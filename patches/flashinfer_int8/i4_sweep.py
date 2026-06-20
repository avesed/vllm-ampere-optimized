#!/usr/bin/env python3
"""I-4a sweep: real per-token int8-QK single_prefill vs fp16 ref across the gate matrix.
PASS per row = cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05]."""
import math, itertools, torch, torch.nn.functional as F
import flashinfer
torch.manual_seed(0)
dev = "cuda"

def run(L, H, HKV, D, causal):
    dt = torch.float16
    q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
    k = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    sm = 1.0 / math.sqrt(D)
    O_ref = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa2", pos_encoding_mode="NONE", sm_scale=sm)
    def qpt(x):
        s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
        xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
        return xi, s.to(torch.float32)
    k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
    q_i8, sq = qpt(q); k_i8, sk = qpt(k_sm)
    sv = v.float().abs().amax() / 127.0
    v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
    O = flashinfer.single_prefill_with_kv_cache(
        q_i8, k_i8, v_i8, scale_q=sq, scale_k=sk, causal=causal, backend="fa2",
        o_dtype=torch.float16, pos_encoding_mode="NONE", sm_scale=sm) * sv
    a, b = O.flatten().float(), O_ref.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
    ratio = (a[fin].norm() / b[fin].norm()).item()
    nan = bool(torch.isnan(O).any())
    return cos, ratio, nan

rows = []
for D in (128, 256):
    for L in (256, 2048):
        for causal in (True, False):
            for (H, HKV) in ((8, 8), (8, 2)):  # MHA and GQA g4
                cos, ratio, nan = run(L, H, HKV, D, causal)
                ok = (cos > 0.99) and (0.95 <= ratio <= 1.05) and not nan
                rows.append((D, L, causal, H, HKV, cos, ratio, ok))
                print(f"D={D} L={L:5d} causal={int(causal)} H={H} HKV={HKV} | "
                      f"cos={cos:.5f} mag={ratio:.4f} {'PASS' if ok else 'FAIL'}")
allpass = all(r[-1] for r in rows)
print("\nSWEEP:", "ALL_PASS" if allpass else f"{sum(r[-1] for r in rows)}/{len(rows)} pass")
