# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere VERIFY leg — MTP spec-decode verify (uniform q=1+K) via famp's OWN vendored XQA
kernel (flashampere/xqa). XQA is a decode-shaped, KV-split, warp-specialized attention whose
q-scaling is ~flat, so q=1+K verify is ~4x faster than the FA2 fwd_kvcache verify on Ampere at
32-128k (cos=1.0, accept-len identical).

CUDAGRAPH-SAFE: the kernel + its launch are capturable (verified: capture+replay cos=1.0, incl.
new-input). To keep the whole leg capturable we (1) call module.xqa_wrapper DIRECTLY (no per-call
Python alloc/decorator overhead — the FI xqa() wrapper would alloc scale tensors each call), (2)
use PERSISTENT buffers (module, scratch, per-(batch,q_seq_len) semaphore/seq_lens-u32/mask), built
lazily on the first (eager warmup) call and reused inside the captured graph, (3) use only
capturable ops in the hot path (copy_, zero_, the kernel) — no .item()/.tolist()/new allocations.
"""
from __future__ import annotations

import math

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends import flash_attn as _fa

from .dispatch import KernelDecline

logger = init_logger(__name__)


def _round_up(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


# Persistent state (cudagraph-safe: allocated/JIT-built once on warmup, reused in the captured graph).
_MODULES: dict[tuple, object] = {}        # (dtype,kv_dtype,page_size,head_dim,grp,q_seq_len) -> module
_BUFS: dict[tuple, tuple] = {}            # (batch,q_seq_len,Hkv,dev) -> (sem, seq_u32, mask)
_SCRATCH: torch.Tensor | None = None
_SM_COUNT: int | None = None
_FIRED = False


def _scratch(device: torch.device) -> torch.Tensor:
    global _SCRATCH
    if _SCRATCH is None or _SCRATCH.device != device:
        _SCRATCH = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
    return _SCRATCH


def _sm_count(device: torch.device) -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT


def _module(input_dtype, kv_dtype, page_size, head_dim, grp, q_seq_len):
    key = (input_dtype, kv_dtype, page_size, head_dim, grp, q_seq_len)
    mod = _MODULES.get(key)
    if mod is None:
        # famp's OWN builder (vendored csrc, Ampere-un-gated). build_and_load() = nvcc on first use;
        # happens on the eager warmup call, BEFORE cudagraph capture (cannot compile during capture).
        from .xqa._jit_xqa import gen_xqa_module
        mod = gen_xqa_module(
            input_dtype, kv_dtype, page_size, head_dim, grp, False, input_dtype, q_seq_len
        ).build_and_load()
        _MODULES[key] = mod
    return mod


def _buffers(batch: int, q_seq_len: int, Hkv: int, device: torch.device):
    key = (batch, q_seq_len, Hkv, device)
    bufs = _BUFS.get(key)
    if bufs is None:
        nb_seq = Hkv * batch
        sem = torch.zeros(_round_up(nb_seq, 2) + 2 + nb_seq + 2, dtype=torch.uint32, device=device)
        seq_u32 = torch.zeros(batch, 1, dtype=torch.uint32, device=device)
        # Packed causal mask [batch, q_seq_len, divUp(q_seq_len,32)*2] uint16 (lower-tri over the
        # last q_seq_len KV positions == FA2 causal q=1+K). Static (depends only on batch,q_seq_len).
        npm = (q_seq_len + 31) // 32
        qi = torch.arange(q_seq_len, device=device).unsqueeze(1)
        ki = torch.arange(q_seq_len, device=device).unsqueeze(0)
        bm = ki <= qi
        pad = npm * 32
        if pad > q_seq_len:
            bm = torch.cat(
                [bm, torch.zeros(q_seq_len, pad - q_seq_len, device=device, dtype=torch.bool)], 1
            )
        bm = bm.view(q_seq_len, npm, 32)
        bits = torch.tensor([1 << i for i in range(32)], device=device, dtype=torch.int64)
        mask = (
            (bm.to(torch.int64) * bits).sum(-1).to(torch.uint32)
            .unsqueeze(0).expand(batch, q_seq_len, npm).contiguous().view(torch.uint16)
        )
        bufs = (sem, seq_u32, mask)
        _BUFS[key] = bufs
    return bufs


def is_supported_head_dim(head_dim: int) -> bool:
    return head_dim in (64, 128, 256)  # XQA headElems (mha.h static_assert)


def xqa_verify(impl, layer, query, key, value, kv_cache, m, output):
    """MTP verify (uniform q=1+K) through famp's vendored XQA. CUDAGRAPH-SAFE."""
    num_reqs = m.seq_lens.shape[0]
    q_seq_len = int(m.max_query_len)
    if q_seq_len <= 1 or m.num_actual_tokens != num_reqs * q_seq_len:
        raise KernelDecline  # not a uniform q=1+K verify batch -> sink to base-FA fwd_kvcache

    H, Hkv, D = impl.num_heads, impl.num_kv_heads, impl.head_size
    if D not in (64, 128, 256) or query.dtype not in (torch.float16, torch.bfloat16):
        raise KernelDecline

    key_cache, value_cache = kv_cache.unbind(1)  # [num_blocks, page_size, Hkv, D] (NHD == XQA NHD)
    _fa.reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        m.slot_mapping, impl.kv_cache_dtype, layer._k_scale, layer._v_scale,
    )
    page_size = key_cache.shape[1]
    dev = query.device

    # The module JIT-build (nvcc) + buffer allocation must NOT happen during cudagraph capture. They
    # run on the first EAGER (warmup) call for a shape; if the very first call for a shape is a
    # captured one (no prior warmup), decline -> base-FA verify for this capture, build next eager call.
    mod_key = (query.dtype, key_cache.dtype, page_size, D, H // Hkv, q_seq_len)
    buf_key = (num_reqs, q_seq_len, Hkv, dev)
    if (mod_key not in _MODULES or buf_key not in _BUFS) and torch.cuda.is_current_stream_capturing():
        raise KernelDecline

    try:
        mod = _module(query.dtype, key_cache.dtype, page_size, D, H // Hkv, q_seq_len)
    except Exception as e:  # JIT build / unsupported config -> sink
        raise KernelDecline from e
    sem, seq_u32, mask = _buffers(num_reqs, q_seq_len, Hkv, dev)

    q5 = query.view(num_reqs, 1, q_seq_len, H, D)       # view of the (persistent) query buffer
    out5 = output.view(num_reqs, 1, q_seq_len, H, D)
    # Fixed grid for cudagraph: max_seq_len = block_table capacity (constant per captured shape);
    # the kernel handles actual per-request seq_lens (GPU) <= this.
    max_seq_len = int(m.block_table.shape[1]) * page_size
    q_scale = impl.scale * math.sqrt(D)  # XQA qk_scale = q_scale*kv_scale/sqrt(D) -> impl.scale

    # Capturable hot path: refresh the persistent seq_lens (int32->uint32) + zero the reduction
    # semaphores, then launch. No allocations, no host syncs.
    seq_u32.copy_(m.seq_lens.view(num_reqs, 1))
    sem.zero_()
    mod.xqa_wrapper(
        False,            # run_sm90_fp8_mha
        _sm_count(dev),   # sm_count
        Hkv, 0,           # num_kv_heads, sliding_win_size
        q_scale, None,    # q_scale (float), q_scale tensor
        out5, 1.0,        # output, rcp_out_scale
        q5, None,         # q, sinks
        key_cache, value_cache, None, None,  # k/v cache, k/v sf
        m.block_table, max_seq_len, seq_u32, num_reqs,  # page_table, max_seq_len, seq_lens, batch
        1.0, None,        # kv_scale (float), kv_scale tensor
        q_seq_len, mask,  # q_seq_len, mask
        sem, _scratch(dev), False,  # semaphores, workspace, enable_pdl
    )
    global _FIRED
    if not _FIRED:
        _FIRED = True
        logger.info("FLASHAMPERE xqa_verify FIRED (cudagraph-safe; q_seq_len=%d head_size=%d num_reqs=%d)",
                    q_seq_len, D, num_reqs)
    return output
