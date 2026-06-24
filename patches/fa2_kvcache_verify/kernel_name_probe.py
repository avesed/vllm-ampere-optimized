#!/usr/bin/env python3
"""Name the actual CUDA kernel behind varlen-q3 (slow) vs fwd_kvcache-q3 (fast) at 32k, verify shape."""
import torch
from torch.profiler import ProfilerActivity, profile
from vllm.vllm_flash_attn import flash_attn_varlen_func, _vllm_fa2_C  # noqa: F401

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS, kv, B, ql = 256, 16, 4, 16, 32768, 1, 3
nblk = (kv + BS - 1) // BS
kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)
qv = torch.randn(B * ql, Hq, D, device=dev, dtype=dt)
cu = torch.tensor([0, ql], device=dev, dtype=torch.int32)
qk = torch.randn(B, ql, Hq, D, device=dev, dtype=dt)


def varlen():
    flash_attn_varlen_func(q=qv, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=ql, seqused_k=seqk,
                           max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, block_table=bt)


def kvc():
    op(qk, kc, vc, None, None, seqk, None, None, None, None, bt, None, None,
       D ** -0.5, True, -1, -1, 0.0, False, 0)


for name, fn in [("VARLEN-q3 (slow)", varlen), ("FWD_KVCACHE-q3 (fast)", kvc)]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    print(f"==== {name} ====")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=5))
