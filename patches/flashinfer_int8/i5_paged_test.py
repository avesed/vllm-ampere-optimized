#!/usr/bin/env python3
"""I-5 MULTI-REQUEST paged int8-QK validation (the per-request k-scale offset gate).

N>=3 requests of DIFFERENT lengths, GQA (Hkv<Hqo), each with its OWN per-token int8 q/k quant +
scales. A single BATCHED paged prefill is compared, PER REQUEST, to that request's fp16
single-request reference. A wrong per-request k-scale offset leaves request 0 correct but corrupts
requests 1..N-1 (cos drops / wrong magnitude), so we PRINT cos+mag for EVERY request.

GATE: every request cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05].

Layout: scales are FLAT per-token over the whole batch in logical (request-major) order:
  scale_q : [sum(qo_len_i)]  indexed by q_indptr[req] + local_q   (q_indptr auto-plumbed by wrapper)
  scale_k : [sum(kv_len_i)]  indexed by kv_scale_indptr[req] + local_kv (kv_scale_indptr auto-plumbed)
The caller passes flat scale_q/scale_k exactly like single-request; the wrapper derives the
per-request kv offset internally (mirrors how q_indptr is used for q-scale)."""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", "2"))         # GQA group 4
D = int(os.environ.get("D", "128"))
PAGE = int(os.environ.get("PAGE", "16"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
LENS = [int(x) for x in os.environ.get("LENS", "128,333,777").split(",")]
N = len(LENS)
dt = torch.float16
sm = 1.0 / math.sqrt(D)
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def qpt(x):  # per-token symmetric int8 over (heads,dim); returns int8 + per-token fp32 scale
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)


# ---- build N requests; keep per-request q/k/v, int8, scales, and a paged copy ----
# CRITICAL: give each request a DISTINCT K magnitude so its per-token k_scale differs strongly
# across requests. With similar scales a wrong k-scale offset only nicks cos ~1e-4 (false PASS);
# with distinct per-request K mags, reading request-0's k_scale for request-i visibly corrupts
# the logit magnitude => softmax distribution => cos/mag drop hard. This is the real gate.
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

# ---- per-request fp16 single references (one prefill each) ----
for r in reqs:
    L = r["L"]
    qo_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=CAUSAL,
            q_data_type=dt, kv_data_type=dt, o_data_type=dt, pos_encoding_mode="NONE", sm_scale=sm)
    r["ref"] = wr.run(r["q"], r["k"], r["v"])

# ---- one BATCHED paged int8 prefill over all N requests ----
# paged layout NHD: [num_pages, PAGE, HKV, D]; each request gets ceil(L/PAGE) pages, concatenated.
pages_per = [(r["L"] + PAGE - 1) // PAGE for r in reqs]
last_page = [r["L"] - (pages_per[i] - 1) * PAGE for i, r in enumerate(reqs)]
total_pages = sum(pages_per)
kpage = torch.zeros(total_pages, PAGE, HKV, D, device=dev, dtype=torch.int8)
vpage = torch.zeros(total_pages, PAGE, HKV, D, device=dev, dtype=torch.int8)
pg = 0
for i, r in enumerate(reqs):
    L = r["L"]
    np_i = pages_per[i]
    pad = np_i * PAGE - L
    ki = r["k_i8"]
    vi = r["v_i8"]
    if pad:
        ki = torch.cat([ki, torch.zeros(pad, HKV, D, device=dev, dtype=torch.int8)], 0)
        vi = torch.cat([vi, torch.zeros(pad, HKV, D, device=dev, dtype=torch.int8)], 0)
    kpage[pg:pg + np_i] = ki.view(np_i, PAGE, HKV, D)
    vpage[pg:pg + np_i] = vi.view(np_i, PAGE, HKV, D)
    pg += np_i

qo_indptr = torch.tensor([0] + list(torch.tensor([r["L"] for r in reqs]).cumsum(0).tolist()),
                         dtype=torch.int32, device=dev)
kv_indptr = torch.tensor([0] + list(torch.tensor(pages_per).cumsum(0).tolist()),
                         dtype=torch.int32, device=dev)
kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
kv_last = torch.tensor(last_page, dtype=torch.int32, device=dev)

q_i8_cat = torch.cat([r["q_i8"] for r in reqs], 0)            # [sum L, H, D]
sq_cat = torch.cat([r["sq"] for r in reqs], 0)                # [sum L]
sk_cat = torch.cat([r["sk"] for r in reqs], 0)               # [sum kv_len] logical order

print(f"N={N} LENS={LENS} H={H} HKV={HKV} D={D} page={PAGE} causal={CAUSAL}")
print(f"qo_indptr={qo_indptr.tolist()} kv_indptr(pages)={kv_indptr.tolist()} last={last_page}")

wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wr.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, PAGE,
        causal=CAUSAL, q_data_type=torch.int8, kv_data_type=torch.int8,
        o_data_type=torch.float16, pos_encoding_mode="NONE", sm_scale=sm)
O_cat = wr.run(q_i8_cat, (kpage, vpage), sq_cat, sk_cat)      # [sum L, H, D] (pre-V-scale)

# ---- per-request compare (apply that request's sv to its output slice) ----
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
print("RESULT:", "I5_PAGED_PASS" if allok else "I5_PAGED_FAIL")
