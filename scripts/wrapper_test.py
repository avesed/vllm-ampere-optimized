#!/usr/bin/env python3
"""Validate the SHIPPED wrapper flash_attn_kvcache_verify (in-place out + arity self-check) against
flash_attn_varlen_func at the forward call shape (flat q=[N,H,D], in-place out)."""
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.vllm_flash_attn import flash_attn_kvcache_verify, flash_attn_varlen_func

print("env VLLM_FA2_KVCACHE_VERIFY default:", envs.VLLM_FA2_KVCACHE_VERIFY)
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS = 256, 16, 4, 16
torch.manual_seed(0)


def test(B, ql, kv):
    nblk = (kv + BS - 1) // BS
    kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
    seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)
    qflat = torch.randn(B * ql, Hq, D, device=dev, dtype=dt)
    ov = torch.empty_like(qflat)
    cu = torch.arange(0, B * ql + 1, ql, device=dev, dtype=torch.int32)
    flash_attn_varlen_func(q=qflat, k=kc, v=vc, out=ov, cu_seqlens_q=cu, max_seqlen_q=ql,
                           seqused_k=seqk, max_seqlen_k=kv, softmax_scale=D ** -0.5,
                           causal=True, block_table=bt)
    ok = torch.empty_like(qflat)  # in-place target, as in forward (output[:N])
    flash_attn_kvcache_verify(q=qflat, k_cache=kc, v_cache=vc, out=ok, seqlens_k=seqk,
                              block_table=bt, num_reqs=B, q_len=ql, softmax_scale=D ** -0.5,
                              causal=True)
    return F.cosine_similarity(ov.float().flatten(), ok.float().flatten(), dim=0).item()


bad = 0
for kv in [4096, 16384, 32768]:
    for ql in [1, 3]:
        cos = test(1, ql, kv)
        ok = cos > 0.9999
        bad += not ok
        print(f"kv={kv:6d} q={ql}: wrapper-vs-varlen cos={cos:.6f}  {'OK' if ok else '*** FAIL ***'}")
print("ALL PASS" if bad == 0 else f"{bad} FAILED")
