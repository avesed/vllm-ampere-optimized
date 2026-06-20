#!/usr/bin/env python3
"""Perf: fp16 vs int8-QK single_prefill with REAL per-token scales (production path).
Confirms the per-token scale reads don't regress the I-2 op-level speedup (~1.05-1.16x)."""
import math, os, torch, flashinfer
torch.manual_seed(0); dev = "cuda"
D = int(os.environ.get("D", "128")); L = int(os.environ.get("L", "16384")); H = 8
sm = 1.0 / math.sqrt(D)

def bench(fn, w=3, it=9):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

q = torch.randn(L, H, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(L, H, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(L, H, D, device=dev, dtype=torch.float16) * 0.5
f16 = lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2",
                                                      pos_encoding_mode="NONE", sm_scale=sm)
def qpt(x):
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    return torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8), s.to(torch.float32)
ks = (k.float() - k.float().mean(0, keepdim=True)).to(torch.float16)
qi, sq = qpt(q); ki, sk = qpt(ks)
sv = v.float().abs().amax() / 127.0
vi = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
i8 = lambda: flashinfer.single_prefill_with_kv_cache(qi, ki, vi, scale_q=sq, scale_k=sk, causal=True,
                                                     backend="fa2", o_dtype=torch.float16,
                                                     pos_encoding_mode="NONE", sm_scale=sm)
f16(); i8()
t16 = bench(f16); ti8 = bench(i8)
print(f"D={D} L={L}  fp16={t16:.3f}ms  int8(per-tok)={ti8:.3f}ms  speedup={t16/ti8:.3f}x", flush=True)
