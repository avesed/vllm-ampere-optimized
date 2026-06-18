#!/usr/bin/env python3
"""I-3 ragged sweep: int8-QK BatchPrefillWithRaggedKVCacheWrapper vs fp16 ragged ref.
PASS = cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05]. Single request per call."""
import math, torch, torch.nn.functional as F, flashinfer
torch.manual_seed(0); dev = "cuda"
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

def run(L, H, HKV, D, causal):
    dt = torch.float16; sm = 1.0 / math.sqrt(D)
    q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
    k = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    qo = torch.tensor([0, L], dtype=torch.int32, device=dev)
    kvp = torch.tensor([0, L], dtype=torch.int32, device=dev)
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    wr.plan(qo, kvp, H, HKV, D, causal=causal, q_data_type=dt, kv_data_type=dt, o_data_type=dt,
            pos_encoding_mode="NONE", sm_scale=sm)
    O_ref = wr.run(q, k, v)
    def qpt(x):
        s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
        return torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8), s.to(torch.float32)
    k_sm = (k.float() - k.float().mean(0, keepdim=True)).to(dt)
    q_i8, sq = qpt(q); k_i8, sk = qpt(k_sm)
    sv = v.float().abs().amax() / 127.0
    v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
    wi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    wi.plan(qo, kvp, H, HKV, D, causal=causal, q_data_type=torch.int8, kv_data_type=torch.int8,
            o_data_type=torch.float16, pos_encoding_mode="NONE", sm_scale=sm)
    O = wi.run(q_i8, k_i8, v_i8, sq, sk) * sv
    a, b = O.flatten().float(), O_ref.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    return F.cosine_similarity(a[fin], b[fin], dim=0).item(), (a[fin].norm() / b[fin].norm()).item(), bool(torch.isnan(O).any())

cases = [(256,8,8,128,True),(256,8,2,128,True),(256,8,8,128,False),(64,8,8,128,True),
         (16,8,8,128,False),(2048,8,8,128,True),(2048,8,2,128,True),
         (256,8,8,256,True),(256,8,2,256,True),(2048,8,8,256,True),(333,8,2,256,True)]
rows = []
for (L, H, HKV, D, causal) in cases:
    cos, ratio, nan = run(L, H, HKV, D, causal)
    ok = (cos > 0.99) and (0.95 <= ratio <= 1.05) and not nan
    rows.append(ok)
    print(f"L={L:5d} H={H} HKV={HKV} D={D} causal={int(causal)} | cos={cos:.5f} mag={ratio:.4f} {'PASS' if ok else 'FAIL'}")
print("\nRAGGED_SWEEP:", "ALL_PASS" if all(rows) else f"{sum(rows)}/{len(rows)} pass")
