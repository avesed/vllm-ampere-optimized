#!/usr/bin/env python3
"""Does long context make the verify compute-bound? Sweep s (context) at fixed q. Attention
arithmetic intensity = 8*(1+K) FLOPs/byte is s-INDEPENDENT (compute ~k*s, KV read ~s, both scale
with s), so the binding resource shouldn't change with s. Report wall vs the KV-byte-read bound
(memory) and the FLOP bound (compute) at each s; whichever the wall tracks at large s is the
binding constraint. If memory at all s -> int8-COMPUTE never helps even long-ctx (fp8-KV-bytes would)."""
import time

import torch
from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS, B = 256, 16, 4, 16, 1
TFLOPS, GBs = 70e12, 936e9  # 3090 bf16 tensor / HBM peak
torch.manual_seed(0)


def bench(ql, kv, iters=50):
    nblk = (kv + BS - 1) // BS
    q = torch.randn(B, ql, Hq, D, device=dev, dtype=dt)
    kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
    bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
    seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)
    out = torch.empty(B, ql, Hq, D, device=dev, dtype=dt)
    def run():
        op(q, kc, vc, None, None, seqk, None, None, None, None, bt, None, out,
           D ** -0.5, True, -1, -1, 0.0, False, 0)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3


for ql in [3, 8]:
    print(f"\n=== q = {ql} (K={ql - 1}), arithmetic intensity = {8 * ql} FLOP/byte (ridge ~75) ===")
    print(f"{'ctx s':>8} {'wall(ms)':>9} {'mem-bound':>10} {'flop-bound':>11} {'binds on':>10}")
    for kv in [8192, 16384, 32768, 65536, 131072]:
        ms = bench(ql, kv)
        mem = kv * Hkv * D * 2 * 2 / GBs * 1e3   # K+V bytes / bw, ms
        flop = ql * Hq * kv * D * 2 * 2 / TFLOPS * 1e3
        binds = "compute" if flop > mem else "memory"
        print(f"{kv:>8} {ms:>9.4f} {mem:>10.4f} {flop:>11.4f} {binds:>10}")
