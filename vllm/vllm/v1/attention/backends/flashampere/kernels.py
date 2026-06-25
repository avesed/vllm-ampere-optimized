# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere prefill kernel — the fp16-PV FlashInfer prefill leg.

fp16pv_prefill() runs a fp16 FlashInfer prefill with use_fp16_pv_reduction (the patch-0007
half-accumulate PV — the measured Ampere prefill win on GeForce-GA10x). It serves both fp16-served
models (leg="fp16pv") and bf16-served models (leg="bf16cvt", runtime bf16->fp16 upcast of Q/K/V,
lossless in fp16 range). int8-QK was removed: a sweep found it net-negative in every scenario
(fresh 16/32/64k + cached-prefix) — its O(L^2) per-token dequant + fp16-KV-read-then-requant tax
always exceeded the ~1.7% IMMA-QK gain.

Targets hd256 plain-decoder prefill (fresh + cached-prefix chunks); the dispatcher guarantees it
is only ever called for a positively-routed PREFILL key, and any run-time shortfall raises
KernelDecline to fall back to stock FA (bit-faithful).
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends import flash_attn as _fa

from .dispatch import KernelDecline

logger = init_logger(__name__)

try:
    import flashinfer  # noqa: F401

    _HAS_FI = True
except Exception as e:  # pragma: no cover - import guard
    flashinfer = None
    _HAS_FI = False
    logger.warning("flashinfer not importable (%s); flashampere prefill legs delegate to FA.", e)

# Fire counter (read by the validation harness): totals + per-leg breakdown.
FIRE = {"calls": 0, "seqs": 0, "fresh": 0, "cached": 0, "fp16pv": 0, "bf16cvt": 0}

# bf16->fp16 upcast is lossless for attention inputs in the fp16 NORMAL range (fp16 carries 10
# mantissa bits vs bf16's 7), but bf16's wider exponent can hold values above fp16 max (65504).
# Such a value would overflow to +-inf on .to(fp16) and NaN the attention. Real post-norm Q/K/V
# are O(1-50) so this never fires in practice, but the bf16cvt leg range-checks all three operands
# before the upcast and declines to bit-faithful FA if any exceeds this -> never emits inf/NaN.
_FP16_MAX = 65504.0

# Per-PROCESS one-time "fired here" latch -> a per-rank log line under TP/PP (separate procs).
_FIRED_LOGGED = False

is_quantized_kv_cache = _fa.is_quantized_kv_cache


def _reshape_and_cache_flash(*args, **kwargs):
    # Bound lazily (FA's varlen kernel binds reshape_and_cache_flash only on a real CUDA build).
    return _fa.reshape_and_cache_flash(*args, **kwargs)


def _gather_one(cache: torch.Tensor, blk: torch.Tensor, n_blocks: int, seq_len: int) -> torch.Tensor:
    """Gather the first `seq_len` logical tokens of ONE request's K or V from the paged cache.
    cache: [num_blocks, block_size, Hkv, D]; blk: [n_blocks] physical block ids. Returns a fresh
    contiguous [seq_len, Hkv, D] in the cache dtype."""
    bs, Hkv, D = cache.shape[1], cache.shape[2], cache.shape[3]
    blocks = cache.index_select(0, blk)
    return blocks.reshape(n_blocks * bs, Hkv, D)[:seq_len]


def _fi_prefill(q, k, v, *, causal, sm, o_dtype, use_fp16_pv=False):
    """Single FlashInfer fp16 prefill; thread use_fp16_pv_reduction only when requested (the param
    exists only on the patch-0007 flashinfer, and the caller gates that via caps.has_fp16pv_kernel)."""
    kw = {"use_fp16_pv_reduction": True} if use_fp16_pv else {}
    return flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa2", o_dtype=o_dtype,
        pos_encoding_mode="NONE", sm_scale=sm, **kw,
    )


def _log_fired_once(leg: str, Lq: int, ctx: int, D: int) -> None:
    global _FIRED_LOGGED
    if _FIRED_LOGGED:
        return
    _FIRED_LOGGED = True
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        rank = f"tp{get_tp_group().rank_in_group}/pp{get_pp_group().rank_in_group}"
    except Exception:
        rank = "single"
    logger.info(
        "FLASHAMPERE %s FIRED rank=%s pid=%d (q_len=%d ctx=%d head_size=%d)",
        leg, rank, os.getpid(), Lq, ctx, D,
    )


def fp16pv_prefill(impl, layer, query, key, value, kv_cache, m, output, *, leg: str = "fp16pv"):
    """fp16-PV prefill — gather/loop + fp16 FlashInfer with use_fp16_pv_reduction. Casts Q/K/V to fp16
    and runs FlashInfer with use_fp16_pv_reduction (DTypeProb=half). Serves BOTH:
      - leg="fp16pv": fp16-served models (Q already fp16; cast is a no-op).
      - leg="bf16cvt": bf16-served models, runtime-upcast bf16->fp16 (lossless in fp16 range) so
        the half-only fp16-PV win reaches bf16 deploys without int8-QK's quant/gather tax.
    Output is written back at output.dtype (fp16->bf16 downcast for bf16 models). Real post-norm
    Q/K/V are O(1-50), far inside fp16 range; but a bf16 source could in principle hold a value
    above fp16-max (65504) that would become inf at the .to(fp16) cast (the cast happens before the
    matmul, so fp32 QK^T accumulation cannot rescue an already-inf operand). For the bf16cvt leg we
    therefore range-check ALL THREE operands before upcast and decline to bit-faithful stock FA if
    any exceeds fp16-max (never fires in practice; pure insurance)."""
    if not _HAS_FI:
        raise KernelDecline
    guard_fp16 = leg == "bf16cvt"  # bf16 source: range-check Q/K/V before upcast (else inf on cast)
    cu_list = m.query_start_loc.tolist()
    n_req = len(cu_list) - 1
    qlens = [int(cu_list[i + 1] - cu_list[i]) for i in range(n_req)]
    seq_lens_cpu = m.seq_lens.tolist()
    for i in range(n_req):
        if qlens[i] == 1 and seq_lens_cpu[i] > 1:
            raise KernelDecline

    key_cache, value_cache = kv_cache.unbind(1)
    _reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        m.slot_mapping, impl.kv_cache_dtype, layer._k_scale, layer._v_scale,
    )
    H, Hkv, D = impl.num_heads, impl.num_kv_heads, impl.head_size
    sm = impl.scale
    causal = m.causal
    block_table = m.block_table
    bs = key_cache.shape[1]

    for i in range(n_req):
        s, e = int(cu_list[i]), int(cu_list[i + 1])
        if e <= s:
            continue
        Lq = e - s
        seq_len = int(seq_lens_cpu[i])
        ctx = seq_len - Lq
        q_src = query[s:e].reshape(Lq, H, D)
        if ctx <= 0:
            k_src = key[s:e].reshape(Lq, Hkv, D)
            v_src = value[s:e].reshape(Lq, Hkv, D)
            FIRE["fresh"] += 1
        else:
            n_blocks = (seq_len + bs - 1) // bs
            blk = block_table[i][:n_blocks]
            k_src = _gather_one(key_cache, blk, n_blocks, seq_len)
            v_src = _gather_one(value_cache, blk, n_blocks, seq_len)
            FIRE["cached"] += 1
        # bf16 source: any Q/K/V element above fp16-max would become inf on the upcast (Q/K turn
        # the score inf, V poisons the PV accumulator) -> range-check all three and sink to
        # bit-faithful FA. The `not (mx <= max)` form also catches a pre-existing NaN/inf operand
        # (NaN fails the <=), so the leg never forwards a non-finite tensor into the cubin. Never
        # fires for real O(1-50) post-norm activations; pure insurance.
        if guard_fp16:
            mx = max(float(q_src.abs().amax()), float(k_src.abs().amax()), float(v_src.abs().amax()))
            if not (mx <= _FP16_MAX):
                raise KernelDecline
        q = q_src.to(torch.float16)
        k = k_src.to(torch.float16)
        v = v_src.to(torch.float16)
        o = _fi_prefill(q, k, v, causal=causal, sm=sm, o_dtype=torch.float16, use_fp16_pv=True)
        output[s:e] = o.reshape(output[s:e].shape).to(output.dtype)
        FIRE["calls"] += 1
        FIRE["seqs"] += 1
        FIRE[leg] += 1
        _log_fired_once(leg, Lq, ctx, D)
    return output
