#!/usr/bin/env python3
"""Time fp16 vs int8 single_prefill for the active compute_qk VARIANT (perturbation profiling)."""
import math, os, torch, flashinfer
torch.manual_seed(0); dev = "cuda"
V = os.environ.get("VARIANT", "full"); D = int(os.environ.get("D", "256")); L = int(os.environ.get("L", "16384")); H = 8

def qpt(x):
    sc = x.abs().amax() / 127.0
    return torch.clamp(torch.round(x / sc), -127, 127).to(torch.int8), float(sc)

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
sm = 1.0 / math.sqrt(D)
f16 = lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2", pos_encoding_mode="NONE", sm_scale=sm)
km = k.float().mean(0, keepdim=True); ks = (k.float() - km).to(torch.float16)
qi, sq = qpt(q.float()); ki, sk = qpt(ks); vi, sv = qpt(v.float()); smi = sm * sq * sk * 256.0
i8 = lambda: flashinfer.single_prefill_with_kv_cache(qi, ki, vi, causal=True, backend="fa2", o_dtype=torch.float16, pos_encoding_mode="NONE", sm_scale=smi)
f16(); i8()
t16 = bench(f16); ti8 = bench(i8)
print(f"VARIANT={V} D={D} L={L}  fp16={t16:.3f}ms  int8={ti8:.3f}ms  int8/fp16={ti8/t16:.3f}  speedup={t16/ti8:.3f}x", flush=True)
