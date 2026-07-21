#!/usr/bin/env python3
"""Where does the verify (fwd_kvcache q=1+K) flip from tensor-core M-starved (q<16, int8 useless)
to FLOP/compute-bound (q>=16, int8 helps)? Sweep q @32k, report wall, per-q marginal, and the
ratio vs the bf16 tensor FLOP bound. If per-q marginal stays ~flat and >>FLOP-bound until q~16,
int8 is NO-GO for realistic MTP K (2-4); if it saturates early, reopen int8."""
import time

import torch
from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS, kv, B = 256, 16, 4, 16, 32768, 1
TFLOPS = 70e12  # ~3090 bf16 tensor (fp32 accumulate)
torch.manual_seed(0)
nblk = (kv + BS - 1) // BS
kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)


def bench(ql, iters=50):
    q = torch.randn(B, ql, Hq, D, device=dev, dtype=dt)
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
    return (time.time() - t) / iters * 1e3  # ms


prev = None
print(f"{'q(=1+K)':>8} {'wall(ms)':>9} {'per-q marg':>11} {'FLOP-bound(ms)':>15} {'wall/FLOP':>10} {'M-util':>7}")
for ql in [1, 2, 3, 4, 6, 8, 16, 32, 64]:
    ms = bench(ql)
    flop = ql * Hq * kv * D * 2 * 2 / TFLOPS * 1e3  # QK+PV, ms
    marg = "" if prev is None else f"{(ms - prev[1]) / (ql - prev[0]):.4f}"
    mutil = f"{min(ql, 16) / 16 * 100:.0f}%"
    print(f"{ql:>8} {ms:>9.4f} {marg:>11} {flop:>15.4f} {ms / flop:>10.1f}x {mutil:>7}")
    prev = (ql, ms)
