#!/usr/bin/env python3
"""Correctness gate: does fwd_kvcache reproduce flash_attn_varlen_func for the verify shape?
Same paged KV, same q, cache_seqlens=full (new verify tokens already written to cache by
reshape_and_cache before attn), causal=True. If cos~=1, fwd_kvcache is a drop-in for the verify."""
import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func, _vllm_fa2_C  # noqa: F401

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS = 256, 16, 4, 16
torch.manual_seed(0)


def test(B, ql, kv):
    nblk = (kv + BS - 1) // BS
    kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
    seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)
    qd = torch.randn(B, ql, Hq, D, device=dev, dtype=dt)
    # varlen: packed q [B*ql, H, D]
    qv = qd.reshape(B * ql, Hq, D)
    cu = torch.arange(0, B * ql + 1, ql, device=dev, dtype=torch.int32)
    out_v = torch.empty_like(qv)
    flash_attn_varlen_func(q=qv, k=kc, v=vc, out=out_v, cu_seqlens_q=cu, max_seqlen_q=ql,
                           seqused_k=seqk, max_seqlen_k=kv, softmax_scale=D ** -0.5,
                           causal=True, block_table=bt)
    # fwd_kvcache: q [B, ql, H, D]; k/v=None (verify tokens already in cache); causal
    res = op(qd, kc, vc, None, None, seqk, None, None, None, None, bt, None, None,
             D ** -0.5, True, -1, -1, 0.0, False, 0)
    out_k = (res[0] if isinstance(res, (list, tuple)) else res).reshape(B, ql, Hq, D)
    a = out_v.reshape(B, ql, Hq, D).float()
    b = out_k.float()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return cos, (a - b).abs().max().item(), a.abs().mean().item()


for kv in [4096, 16384, 32768]:
    for ql in [1, 3]:
        cos, md, scale = test(1, ql, kv)
        flag = "OK" if cos > 0.999 else "*** MISMATCH ***"
        print(f"kv={kv:6d} q={ql}: cos={cos:.6f}  maxdiff={md:.4f}  (scale~{scale:.3f})  {flag}", flush=True)
