#!/usr/bin/env python3
"""I-5 MULTI-REQUEST ragged int8-QK validation (the per-request k-scale offset gate).

Same contract as i5_paged_test.py but for BatchPrefillWithRaggedKVCacheWrapper: N>=3 requests of
different lengths, GQA, each with its OWN per-token int8 q/k quant + scales, ONE batched ragged
prefill, compared PER REQUEST to that request's fp16 single ragged reference.

GATE: every request cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05].
A wrong per-request k-scale offset => req0 OK, req1..N-1 wrong (so we print all)."""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", "2"))
D = int(os.environ.get("D", "128"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
LENS = [int(x) for x in os.environ.get("LENS", "128,333,777").split(",")]
N = len(LENS)
dt = torch.float16
sm = 1.0 / math.sqrt(D)
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def qpt(x):
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)


# CRITICAL: distinct per-request K magnitude (see i5_paged_test.py) so a wrong k-scale offset
# is unmistakable (otherwise statistically-similar scales hide the bug at cos~0.999).
KMAG = [float(x) for x in os.environ.get("KMAG", "0.1,1.0,8.0").split(",")]
reqs = []
for ri, L in enumerate(LENS):
    km = KMAG[ri % len(KMAG)]
    q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
    k = (torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5 * km).to(dt)
    v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
    q_i8, sq = qpt(q)
    k_i8, sk = qpt(k_sm)
    sv = v.float().abs().amax() / 127.0
    v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
    reqs.append(dict(L=L, q=q, k=k, v=v, q_i8=q_i8, sq=sq, k_i8=k_i8, sk=sk,
                     v_i8=v_i8, sv=sv.to(torch.float32)))

# per-request fp16 single ragged references
for r in reqs:
    L = r["L"]
    qo_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=CAUSAL,
            q_data_type=dt, kv_data_type=dt, o_data_type=dt, pos_encoding_mode="NONE", sm_scale=sm)
    r["ref"] = wr.run(r["q"], r["k"], r["v"])

# one BATCHED ragged int8 prefill
q_i8_cat = torch.cat([r["q_i8"] for r in reqs], 0)
k_i8_cat = torch.cat([r["k_i8"] for r in reqs], 0)
v_i8_cat = torch.cat([r["v_i8"] for r in reqs], 0)
sq_cat = torch.cat([r["sq"] for r in reqs], 0)
sk_cat = torch.cat([r["sk"] for r in reqs], 0)
qo_indptr = torch.tensor([0] + list(torch.tensor([r["L"] for r in reqs]).cumsum(0).tolist()),
                         dtype=torch.int32, device=dev)
kv_indptr = qo_indptr.clone()   # kv_len == qo_len per request here

print(f"N={N} LENS={LENS} H={H} HKV={HKV} D={D} causal={CAUSAL}")
print(f"qo_indptr={qo_indptr.tolist()} kv_indptr={kv_indptr.tolist()}")

wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=CAUSAL,
        q_data_type=torch.int8, kv_data_type=torch.int8, o_data_type=torch.float16,
        pos_encoding_mode="NONE", sm_scale=sm)
O_cat = wr.run(q_i8_cat, k_i8_cat, v_i8_cat, sq_cat, sk_cat)

allok = True
for i, r in enumerate(reqs):
    s, e = qo_indptr[i].item(), qo_indptr[i + 1].item()
    Oi = O_cat[s:e].float() * r["sv"]
    a, b = Oi.flatten(), r["ref"].flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
    mag = (a[fin].norm() / b[fin].norm()).item()
    nan = bool(torch.isnan(Oi).any())
    ok = (cos > 0.99) and (0.95 <= mag <= 1.05) and not nan
    allok &= ok
    print(f"  req{i} L={r['L']:>4}  cos={cos:.5f}  mag={mag:.4f}  {'PASS' if ok else 'FAIL'}")
print("RESULT:", "I5_RAGGED_PASS" if allok else "I5_RAGGED_FAIL")
