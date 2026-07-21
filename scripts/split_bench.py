#!/usr/bin/env python3
"""Force-fast-path probe: does forcing num_splits>0 fix the q=1+K verify attention?
Microbenchmark flash_attn_varlen_func at the MTP verify shape (Hq=16,Hkv=4,D=256, paged) for
q_len=1 (decode) vs q_len=3 (verify), num_splits 0(heuristic)/8/16/32, across KV lengths.
If q3/ns0 ~= 33x q1, confirms root cause; if q3/ns>=16 ~= q1, forcing splits is the fix."""
import time

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS = 256, 16, 4, 16
torch.manual_seed(0)


def bench(B, q_len, kv_len, ns, iters=50):
    nblk = (kv_len + BS - 1) // BS
    q = torch.randn(B * q_len, Hq, D, device=dev, dtype=dt)
    kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
    cu_q = torch.arange(0, B * q_len + 1, q_len, device=dev, dtype=torch.int32)
    seqused = torch.full((B,), kv_len, device=dev, dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        flash_attn_varlen_func(
            q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=q_len,
            seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=D ** -0.5,
            causal=True, block_table=bt, num_splits=ns)

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
                ms = bench(1, ql, kv, ns)
                cells.append(f"q{ql}/ns{ns}={ms:.3f}")
            except Exception as e:
                cells.append(f"q{ql}/ns{ns}=ERR({str(e)[:40]})")
    print(f"kv={kv:6d}: " + "  ".join(cells), flush=True)
