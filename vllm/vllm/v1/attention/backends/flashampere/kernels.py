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

import math
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
    # A decode row (q=1, ctx>0) mixed into a prefill batch (continuous batching) is the one shape FI
    # single_prefill can't do (q_len=1 -> max_mma_kv=0). For hd<=256 we DECLINE so stock FA handles
    # the whole mixed batch optimally. For hd>256 (Gemma4 512) FA has NO path (head_size>256 reject),
    # so declining would crash; instead the per-request loop below handles q=1 rows via q-pad-2.
    if impl.head_size <= 256:
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
        if Lq == 1:
            # q=1 row (decode in a mixed hd>256 batch, or a rare 1-token prefill): FI single_prefill
            # rejects q_len=1, so q-pad to 2 and take the LAST row (causal bottom-right -> attends all
            # ctx incl. this step's just-cached K/V). Same q-pad-2 trick as the hd512 decode path.
            o = _fi_prefill(q.repeat(2, 1, 1), k, v, causal=True, sm=sm,
                            o_dtype=torch.float16, use_fp16_pv=True)[1:2]
        else:
            o = _fi_prefill(q, k, v, causal=causal, sm=sm, o_dtype=torch.float16, use_fp16_pv=True)
        output[s:e] = o.reshape(output[s:e].shape).to(output.dtype)
        FIRE["calls"] += 1
        FIRE["seqs"] += 1
        FIRE[leg] += 1
        _log_fired_once(leg, Lq, ctx, D)
    return output


def _decode_hd512_eager(impl, layer, query, key, value, kv_cache, m, output):
    """EAGER hd512 (Gemma4 full-attn) decode — per-request gather + FI prefill (q-pad-2). The
    capturable path (decode_hd512 below) supersedes this; kept as the not-capturing fallback when
    the paged BatchPrefill wrapper is unavailable. Casts the (small) gathered KV to fp16."""
    if not _HAS_FI:
        raise KernelDecline
    key_cache, value_cache = kv_cache.unbind(1)
    _reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        m.slot_mapping, impl.kv_cache_dtype, layer._k_scale, layer._v_scale,
    )
    H, D = impl.num_heads, impl.head_size
    sm = impl.scale
    bs = key_cache.shape[1]
    seq_lens = m.seq_lens.tolist()
    for i in range(len(seq_lens)):
        seq_len = int(seq_lens[i])
        if seq_len <= 0:
            continue
        q_i = query[i].reshape(1, H, D).to(torch.float16)         # [1, H, D]
        n_blocks = (seq_len + bs - 1) // bs
        blk = m.block_table[i][:n_blocks]
        k_i = _gather_one(key_cache, blk, n_blocks, seq_len).to(torch.float16)  # [seq_len, Hkv=1, D]
        v_i = _gather_one(value_cache, blk, n_blocks, seq_len).to(torch.float16)
        q2 = q_i.repeat(2, 1, 1)                                   # [2, H, D]
        o = _fi_prefill(q2, k_i, v_i, causal=True, sm=sm, o_dtype=torch.float16, use_fp16_pv=True)
        output[i] = o[1].reshape(output[i].shape).to(output.dtype)
        FIRE["calls"] += 1
    if seq_lens:
        _log_fired_once("decode_hd512_eager", 1, int(seq_lens[0]) - 1, impl.head_size)
    return output


class _Hd512DecodeState:
    """Persistent cudagraph state for hd512 (Gemma4 full-attn) DECODE via paged BatchPrefill.

    hd512 has no FA decode kernel (head_size<=256) and every FI DECODE kernel (single/Batch decode)
    rejects Gemma4's MQA group_size=16; only BatchPrefill (q_len>=2) runs it. To make that path
    FULL-cudagraph-capturable we mirror vLLM's FI-backend pattern: ONE shared workspace + max-sized
    metadata buffers, a per-batch-size BatchPrefill wrapper binding to buffer slices, PLANNED ONCE at
    worst-case during the eager warmup (dummy_run), then per step the FI metadata (qo/kv indptr +
    indices + last-page-len) is rebuilt with CAPTURABLE GPU ops from block_table/seq_lens and run() is
    replayed inside the captured graph. Validated cos=1.0: plan-once-run-varying-seqlen + in-graph
    metadata rebuild (new pages appearing) both survive capture/replay. bf16 native (no fp16-PV: the
    wrapper path needs no register relaxation and decode is bandwidth-bound, so half-PV buys nothing).

    q is replicated to length 2 per request (qo_indptr=[0,2,4,...]) and the LAST row of each pair is
    the decode (causal -> sits at the true last position, attends all ctx incl. this step's K/V)."""

    def __init__(self, impl, query, kv_cache, m):
        dev = m.seq_lens.device
        self.ps = kv_cache.shape[2]                       # paged block_size
        self.mb = m.block_table.shape[1]                  # max blocks/req (== cdiv(max_model_len,ps))
        self.max_bs = impl._fc_max_num_seqs               # captured at impl __init__ (config-context)
        self.max_pages = self.max_bs * self.mb
        self.Hq, self.Hkv, self.D = impl.num_heads, impl.num_kv_heads, impl.head_size
        self.sm = impl.scale                              # Gemma uses a non-1/sqrt(D) attn scale
        sw = getattr(impl, "sliding_window", None)        # FA stores (left,right); full-attn -> none
        self.window = int(sw[0]) if isinstance(sw, (tuple, list)) and tuple(sw) != (-1, -1) else -1
        self.qdtype, self.kvdtype = query.dtype, kv_cache.dtype
        # Shared workspace + max-sized metadata buffers (per-bs wrappers bind to slices of these).
        # Sized for the WORST-CASE plan (every req full max_model_len -> split-KV work buffer); the
        # q-pad-2 BatchPrefill scheduler needs more than FI's 256MB default at 8k+ ctx.
        self.ws = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=dev)
        self.qo = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=dev)
        self.kvi = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=dev)
        self.kvidx = torch.zeros(self.max_pages, dtype=torch.int32, device=dev)
        self.klp = torch.zeros(self.max_bs, dtype=torch.int32, device=dev)
        self.qpad = torch.zeros(self.max_bs * 2, self.Hq, self.D, dtype=self.qdtype, device=dev)
        # Constants for the capturable metadata rebuild (no host<->device traffic in the graph).
        self.QO_CONST = torch.arange(0, 2 * (self.max_bs + 1), 2, dtype=torch.int32, device=dev)
        self.COL = torch.arange(self.mb, device=dev).unsqueeze(0).expand(self.max_bs, -1).reshape(-1)
        self.kvidx_ext = torch.zeros(self.max_pages + 1, dtype=torch.int32, device=dev)  # +1 trash slot
        self.ZERO1 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.wrappers: dict[int, object] = {}
        # The N hd512 full-attn layers share one block_table/seq_lens -> identical FI metadata. Rebuild
        # it ONCE per step (the first hd512 layer) and let the rest reuse the buffers. id() match is a
        # capture-time Python branch, so only the first layer's rebuild ops land in the captured graph.
        self._first_layer_id: int | None = None

    def _build_meta(self, block_table, seq_lens, B):
        """CAPTURABLE: block_table[:B]+seq_lens[:B] -> kvi/kvidx/klp buffers (no .item()/.tolist()).
        qo is constant ([0,2,..,2B], set once at plan time) so it is not rebuilt here."""
        npg = (seq_lens + self.ps - 1) // self.ps                                   # [B]
        self.kvi[: B + 1].copy_(torch.cat([self.ZERO1, torch.cumsum(npg, 0).to(torch.int32)]))
        self.klp[:B].copy_(seq_lens - (npg - 1) * self.ps)
        flat = block_table.reshape(-1)                                              # [B*mb] physical ids
        col = self.COL[: B * self.mb]
        valid = col < npg.unsqueeze(1).expand(-1, self.mb).reshape(-1)             # [B*mb] in-range pages
        out_pos = torch.cumsum(valid.to(torch.int64), 0) - 1                        # compact dest index
        dest = torch.where(valid, out_pos, torch.full_like(out_pos, self.max_pages))  # invalid -> trash
        self.kvidx_ext.zero_()
        self.kvidx_ext.scatter_(0, dest, flat)
        self.kvidx.copy_(self.kvidx_ext[: self.max_pages])

    def _ensure_wrapper(self, B):
        """Lazily build+plan a per-bs wrapper at WORST case (every req full mb pages). Eager-only
        (called when not capturing); plan sizes the work buffer for the largest seq_len the captured
        graph can later replay, so subsequent smaller-seq runs stay within it."""
        w = self.wrappers.get(B)
        if w is not None:
            return w
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.ws, "NHD", use_cuda_graph=True,
            qo_indptr_buf=self.qo[: B + 1],
            paged_kv_indptr_buf=self.kvi[: B + 1],
            paged_kv_indices_buf=self.kvidx,
            paged_kv_last_page_len_buf=self.klp[:B],
        )
        dev = self.qo.device
        self.qo[: B + 1].copy_(self.QO_CONST[: B + 1])
        self.kvi[: B + 1].copy_(torch.arange(0, (B + 1) * self.mb, self.mb, dtype=torch.int32, device=dev))
        self.kvidx[: B * self.mb].copy_(torch.arange(B * self.mb, dtype=torch.int32, device=dev))
        self.klp[:B].fill_(self.ps)
        w.plan(
            self.qo[: B + 1], self.kvi[: B + 1], self.kvidx, self.klp[:B],
            self.Hq, self.Hkv, self.D, self.ps, causal=False, sm_scale=self.sm,
            window_left=self.window, q_data_type=self.qdtype, kv_data_type=self.kvdtype,
        )
        self.wrappers[B] = w
        return w

    def run(self, impl, layer, query, key, value, kv_cache, m, output):
        key_cache, value_cache = kv_cache.unbind(1)
        _reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            m.slot_mapping, impl.kv_cache_dtype, layer._k_scale, layer._v_scale,
        )
        B = m.seq_lens.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        w = self.wrappers.get(B)
        if w is None:
            if capturing:
                # Not warmed for this size (dummy_run should have covered every capture size) -> the
                # caller sinks to FA, which rejects hd512. Surface loudly rather than corrupt silently.
                raise KernelDecline
            w = self._ensure_wrapper(B)
        if self._first_layer_id is None:
            self._first_layer_id = id(layer)
        # Rebuild the shared FI metadata only on the first hd512 layer of the step (the others reuse it).
        if id(layer) == self._first_layer_id:
            self._build_meta(m.block_table[:B], m.seq_lens, B)
        q = query.reshape(B, self.Hq, self.D)
        qp = self.qpad[: 2 * B].view(B, 2, self.Hq, self.D)
        qp[:, 0].copy_(q)
        qp[:, 1].copy_(q)
        o = w.run(self.qpad[: 2 * B], kv_cache)                # [2B, Hq, D]
        # Take the FIRST row of each q-pad pair (o[0::2]). The pad is q_len=2 only because the FI
        # paged kernel rejects q_len=1 (max_mma_kv=0); attention is non-causal (causal=False) since a
        # decode query attends ALL ctx incl. its own just-cached K/V, so BOTH rows would equal the
        # decode under an actual-sized plan. But the cudagraph plan is sized at WORST case (full
        # max_model_len) and FI's bottom-right alignment then masks the SECOND row to zero at small
        # actual ctx; row 0 (validated cos=1.0) is unaffected by the worst-case plan -> use it.
        output.copy_(o[0::2].reshape(output.shape).to(output.dtype))
        FIRE["calls"] += 1
        return output


class _XqaHd512DecodeState:
    """Persistent cudagraph state for hd512 (Gemma4 full-attn) DECODE via famp's OWN vendored XQA
    kernel (flashampere/xqa), superseding the FlashInfer q-pad-2 BatchPrefill hack. XQA is a real
    decode kernel (q=1, MQA group16, head_dim=512) — no q-pad waste — so it is faster than FI at the
    common ctx (<=~6k; micro-bench 1k 1.40x / 4k 1.17x). head_dim=512 is realised by the
    headElemsQK(512)/headElems(256) split in mha.h (gemm0 runs full-512 QK; gemm1+output cover the
    512 V/output as TWO 256-wide chunks); here the chunks are TWO kernel calls with the 2nd reading
    V cols [256,512) and writing output cols [256,512) via a +256-element pointer offset (the wrapper
    derives strides from the K-cache, so v_cache[...,256:] just moves the base). cos=1.0 validated.

    CUDAGRAPH-SAFE (same recipe as xqa_verify): module JIT-build + per-bs buffers are made lazily on
    the EAGER warmup call (never during capture); the hot path is only reshape_and_cache + copy_ +
    zero_ + two module.xqa_wrapper launches (no alloc / host-sync / .item()). XQA takes block_table +
    seq_lens directly, so there is NO FI-style metadata rebuild. gemm0-once (single K read, a uniform
    win incl. 8k+/large-batch) is the documented next optimisation."""

    def __init__(self, impl, query, kv_cache, m):
        dev = m.seq_lens.device
        self.ps = kv_cache.shape[2]                       # paged block_size
        self.mb = m.block_table.shape[1]                  # max blocks/req
        self.Hq, self.Hkv, self.D = impl.num_heads, impl.num_kv_heads, impl.head_size
        self.grp = self.Hq // self.Hkv
        self.half = self.D // 2                           # 256 — the gemm1/output chunk width
        # XQA qkScale = q_scale * rsqrt(D); we want it == impl.scale (Gemma's custom attn scale).
        self.q_scale = float(impl.scale) * math.sqrt(self.D)
        self.qdtype, self.kvdtype = query.dtype, kv_cache.dtype
        self.sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
        self.scratch = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
        self.module = None
        self.bufs: dict[int, tuple] = {}                  # B -> (sem, seq_u32)

    def _get_module(self):
        if self.module is None:
            from .xqa._jit_xqa import gen_xqa_module
            # full-attn -> no sliding window; output dtype == input dtype; q_seq_len=1 (decode).
            self.module = gen_xqa_module(
                self.qdtype, self.kvdtype, self.ps, self.D, self.grp, False, self.qdtype, 1
            ).build_and_load()
        return self.module

    def _get_bufs(self, B, dev):
        b = self.bufs.get(B)
        if b is None:
            nb_seq = self.Hkv * B
            sem = torch.zeros((nb_seq + 1) // 2 * 2 + 2 + nb_seq + 2, dtype=torch.uint32, device=dev)
            seq_u32 = torch.zeros(B, 1, dtype=torch.uint32, device=dev)
            b = self.bufs[B] = (sem, seq_u32)
        return b

    def run(self, impl, layer, query, key, value, kv_cache, m, output):
        key_cache, value_cache = kv_cache.unbind(1)       # [num_blocks, page_size, Hkv, D] (NHD)
        _reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            m.slot_mapping, impl.kv_cache_dtype, layer._k_scale, layer._v_scale,
        )
        B = m.seq_lens.shape[0]
        dev = query.device
        capturing = torch.cuda.is_current_stream_capturing()
        # JIT build + buffer alloc must NOT run during capture -> decline (caller sinks to FA, which
        # rejects hd512; the eager warmup builds these so the next capture for this size succeeds).
        if (self.module is None or B not in self.bufs) and capturing:
            raise KernelDecline
        mod = self._get_module()
        sem, seq_u32 = self._get_bufs(B, dev)
        q4 = query.view(B, 1, self.Hq, self.D)
        out4 = output.view(B, 1, self.Hq, self.D)
        max_seq_len = self.mb * self.ps
        page_table = m.block_table[:B]
        # Capturable hot path: refresh persistent seq_lens (int32->uint32), then two launches.
        seq_u32.copy_(m.seq_lens.view(B, 1))

        def _launch(out_t, v_t):
            sem.zero_()
            mod.xqa_wrapper(
                False, self.sm_count, self.Hkv, 0,        # run_sm90, sm_count, num_kv_heads, sliding_win
                self.q_scale, None,                       # q_scale (float), q_scale tensor
                out_t, 1.0,                               # output, rcp_out_scale
                q4, None,                                 # q, sinks
                key_cache, v_t, None, None,               # k/v cache, k/v sf
                page_table, max_seq_len, seq_u32, B,      # page_table, max_seq_len, seq_lens, batch
                1.0, None,                                # kv_scale (float), kv_scale tensor
                1, None,                                  # q_seq_len=1 (decode), mask
                sem, self.scratch, False,                 # semaphores, workspace, enable_pdl
            )

        _launch(out4, value_cache)                        # chunk 0: V cols [0,256) -> out cols [0,256)
        _launch(out4[..., self.half:], value_cache[..., self.half:])  # chunk 1: +256 ptr offset
        FIRE["calls"] += 1
        return output


_STATES: dict[int, object] = {}   # keyed by head_size (hd512 full-attn decode state)


def _use_xqa_hd512() -> bool:
    # Opt-in (default off) while the FI BatchPrefill path stays the validated shipped default. Flip to
    # default-on after the e2e (cudagraph + GSM8K) validation of the owned XQA hd512 decode kernel.
    return os.environ.get("VLLM_FAMP_XQA_HD512", "0") == "1"


def decode_hd512(impl, layer, query, key, value, kv_cache, m, output):
    """hd512 (Gemma4 full-attn) DECODE — FULL-cudagraph-capturable paged BatchPrefill (q-pad-2),
    superseding the eager per-request gather. A process-global _Hd512DecodeState holds the shared
    cudagraph buffers + per-bs wrappers (planned at worst-case during warmup); each step rebuilds the
    FI metadata capturably and replays run() in the captured decode graph. Not-capturing calls plan
    on demand (so eager serving works too); a capturing call for an un-warmed batch size declines to
    the eager fallback. See _Hd512DecodeState for the validated capture/replay correctness."""
    if not _HAS_FI:
        raise KernelDecline
    hs = impl.head_size
    try:
        st = _STATES.get(hs)
        if (
            st is None
            and getattr(impl, "_fc_max_num_seqs", None)
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Owned XQA hd512 kernel (faster, drops the FI q-pad-2 hack) is opt-in; FI is the default.
            cls = _XqaHd512DecodeState if _use_xqa_hd512() else _Hd512DecodeState
            st = _STATES[hs] = cls(impl, query, kv_cache, m)
        if st is not None:
            out = st.run(impl, layer, query, key, value, kv_cache, m, output)
            _log_fired_once("decode_hd512", 1, 0, impl.head_size)
            return out
    except KernelDecline:
        if torch.cuda.is_current_stream_capturing():
            raise
    # Not-capturing fallback (state un-init or wrapper build failed): eager gather path.
    return _decode_hd512_eager(impl, layer, query, key, value, kv_cache, m, output)
