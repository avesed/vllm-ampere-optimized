#!/usr/bin/env python3
"""Test the REAL fast path: torch.ops._vllm_fa2_C.fwd_kvcache (compiled but unwrapped) at the MTP
verify shape (q=1+K, Hq=16,Hkv=4,D=256, paged, causal), vs the varlen baseline. fwd_kvcache supports
num_splits on FA2 (varlen does not) -> FlashDecoding KV-split for q>1. If q3 fwd_kvcache ~ q1 speed,
forcing this path is the fix."""
import time

import torch
from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401  (registers the op)

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS = 256, 16, 4, 16
torch.manual_seed(0)


def bench(B, q_len, kv_len, ns, iters=50):
    nblk = (kv_len + BS - 1) // BS
    q = torch.randn(B, q_len, Hq, D, device=dev, dtype=dt)
    kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
    seqk = torch.full((B,), kv_len, device=dev, dtype=torch.int32)

    def run():
        # (q, kcache, vcache, k, v, seqlens_k, rotary_cos, rotary_sin, cache_batch_idx,
        #  leftpad_k, block_table, alibi_slopes, out, softmax_scale, is_causal,
        #  window_size_left, window_size_right, softcap, is_rotary_interleaved, num_splits)
        op(q, kc, vc, None, None, seqk, None, None, None, None, bt, None, None,
           D ** -0.5, True, -1, -1, 0.0, False, ns)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3


for kv in [4096, 16384, 32768]:
    cells = []
    for ql in [1, 3]:
        for ns in [0, 8, 16, 32]:
            try:
                cells.append(f"q{ql}/ns{ns}={bench(1, ql, kv, ns):.3f}")
            except Exception as e:
                cells.append(f"q{ql}/ns{ns}=ERR({str(e)[:45]})")
    print(f"kv={kv:6d}: " + "  ".join(cells), flush=True)
