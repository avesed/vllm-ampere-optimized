#!/usr/bin/env python3
"""I-5 multi-request sweep: paged + ragged, many configs. Each config = ONE batched prefill of N
requests with DISTINCT per-request K magnitudes (so a wrong k-scale offset is unmistakable),
each request compared PER REQUEST to its own fp16 single reference.

Covers: N in {3,5,8}, head_dim {64,128,256}, GQA groups {1,2,4,8}, causal + non-causal,
page sizes {16,32}, and a qo<kv "append/decode-extend" config (q shorter than kv per request).
PASS = EVERY request in EVERY config cos>0.99 AND mag in [0.95,1.05]."""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(1)
dev = "cuda"
dt = torch.float16
workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=dev)


def qpt(x):
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)


def make_reqs(lens, H, HKV, D, kmag, causal, qo_lens=None):
    """qo_lens: optional per-request qo_len < kv_len (append). default qo_len==kv_len."""
    reqs = []
    for i, L in enumerate(lens):
        km = kmag[i % len(kmag)]
        ql = L if qo_lens is None else qo_lens[i]
        q = torch.randn(ql, H, D, device=dev, dtype=dt) * 0.5
        k = (torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5 * km).to(dt)
        v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
        k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
        q_i8, sq = qpt(q)
        k_i8, sk = qpt(k_sm)
        sv = v.float().abs().amax() / 127.0
        v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
        reqs.append(dict(L=L, ql=ql, q=q, k=k, v=v, q_i8=q_i8, sq=sq, k_i8=k_i8,
                         sk=sk, v_i8=v_i8, sv=sv.to(torch.float32)))
    return reqs


def ref_each(reqs, H, HKV, D, causal):
    for r in reqs:
        qo_indptr = torch.tensor([0, r["ql"]], dtype=torch.int32, device=dev)
        kv_indptr = torch.tensor([0, r["L"]], dtype=torch.int32, device=dev)
        wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
        wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=causal,
                q_data_type=dt, kv_data_type=dt, o_data_type=dt, pos_encoding_mode="NONE",
                sm_scale=1.0 / math.sqrt(D))
        r["ref"] = wr.run(r["q"], r["k"], r["v"])


def compare(O_cat, reqs, qo_indptr, tag):
    allok = True
    worst = 1.0
    for i, r in enumerate(reqs):
        s, e = qo_indptr[i].item(), qo_indptr[i + 1].item()
        Oi = O_cat[s:e].float() * r["sv"]
        a, b = Oi.flatten(), r["ref"].flatten().float()
        fin = torch.isfinite(a) & torch.isfinite(b)
        cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
        mag = (a[fin].norm() / b[fin].norm()).item()
        nan = bool(torch.isnan(Oi).any())
        ok = (cos > 0.99) and (0.95 <= mag <= 1.05) and not nan
        worst = min(worst, cos)
        allok &= ok
        if not ok:
            print(f"    {tag} req{i} L={r['L']} ql={r['ql']} cos={cos:.5f} mag={mag:.4f} FAIL")
    print(f"  {tag}: {'ALL_PASS' if allok else 'FAIL'} (worst cos={worst:.5f}, N={len(reqs)})")
    return allok


def run_ragged(lens, H, HKV, D, kmag, causal, qo_lens=None):
    reqs = make_reqs(lens, H, HKV, D, kmag, causal, qo_lens)
    ref_each(reqs, H, HKV, D, causal)
    q_i8 = torch.cat([r["q_i8"] for r in reqs], 0)
    k_i8 = torch.cat([r["k_i8"] for r in reqs], 0)
    v_i8 = torch.cat([r["v_i8"] for r in reqs], 0)
    sq = torch.cat([r["sq"] for r in reqs], 0)
    sk = torch.cat([r["sk"] for r in reqs], 0)
    qo_indptr = torch.tensor([0] + torch.tensor([r["ql"] for r in reqs]).cumsum(0).tolist(),
                             dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0] + torch.tensor([r["L"] for r in reqs]).cumsum(0).tolist(),
                             dtype=torch.int32, device=dev)
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=causal,
            q_data_type=torch.int8, kv_data_type=torch.int8, o_data_type=torch.float16,
            pos_encoding_mode="NONE", sm_scale=1.0 / math.sqrt(D))
    O = wr.run(q_i8, k_i8, v_i8, sq, sk)
    return compare(O, reqs, qo_indptr, f"RAGGED N={len(lens)} H{H}/{HKV} D{D} c{int(causal)}")


def run_paged(lens, H, HKV, D, kmag, causal, page, qo_lens=None):
    reqs = make_reqs(lens, H, HKV, D, kmag, causal, qo_lens)
    ref_each(reqs, H, HKV, D, causal)
    pages_per = [(r["L"] + page - 1) // page for r in reqs]
    last_page = [r["L"] - (pages_per[i] - 1) * page for i, r in enumerate(reqs)]
    total_pages = sum(pages_per)
    kpage = torch.zeros(total_pages, page, HKV, D, device=dev, dtype=torch.int8)
    vpage = torch.zeros(total_pages, page, HKV, D, device=dev, dtype=torch.int8)
    pg = 0
    for i, r in enumerate(reqs):
        np_i = pages_per[i]
        pad = np_i * page - r["L"]
        ki, vi = r["k_i8"], r["v_i8"]
        if pad:
            ki = torch.cat([ki, torch.zeros(pad, HKV, D, device=dev, dtype=torch.int8)], 0)
            vi = torch.cat([vi, torch.zeros(pad, HKV, D, device=dev, dtype=torch.int8)], 0)
        kpage[pg:pg + np_i] = ki.view(np_i, page, HKV, D)
        vpage[pg:pg + np_i] = vi.view(np_i, page, HKV, D)
        pg += np_i
    q_i8 = torch.cat([r["q_i8"] for r in reqs], 0)
    sq = torch.cat([r["sq"] for r in reqs], 0)
    sk = torch.cat([r["sk"] for r in reqs], 0)
    qo_indptr = torch.tensor([0] + torch.tensor([r["ql"] for r in reqs]).cumsum(0).tolist(),
                             dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0] + torch.tensor(pages_per).cumsum(0).tolist(),
                             dtype=torch.int32, device=dev)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
    kv_last = torch.tensor(last_page, dtype=torch.int32, device=dev)
    wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wr.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, page,
            causal=causal, q_data_type=torch.int8, kv_data_type=torch.int8,
            o_data_type=torch.float16, pos_encoding_mode="NONE", sm_scale=1.0 / math.sqrt(D))
    O = wr.run(q_i8, (kpage, vpage), sq, sk)
    return compare(O, reqs, qo_indptr, f"PAGED  N={len(lens)} H{H}/{HKV} D{D} c{int(causal)} p{page}")


KM = [0.05, 1.0, 12.0, 0.3, 4.0, 0.5, 2.0, 8.0]   # distinct per-request K magnitudes
# head_dim 64 is GUARDED unsupported for int8 (k64B smem swizzle unimplemented) -> tested
# separately below as a must-raise, NOT in the numeric configs (only deployed hd128/hd256).
configs = [
    # (lens, H, HKV, D, causal, page)
    ([128, 333, 777], 8, 2, 128, True, 16),
    ([128, 333, 777], 8, 2, 128, False, 16),
    ([200, 500, 100, 333, 777], 8, 1, 128, True, 32),     # N=5, GQA g8, Hkv=1
    ([64, 128, 256, 512, 333, 200, 99, 450], 16, 16, 128, True, 16),  # N=8 MHA
    ([257, 130, 600], 8, 4, 256, True, 32),               # D256 GQA g2
    ([257, 130, 600], 8, 4, 256, False, 16),              # D256 non-causal
    ([512, 1024, 333], 32, 4, 128, True, 16),             # N=3 GQA g8 larger
]
allok = True
print("=== PAGED ===")
for lens, H, HKV, D, c, pg in configs:
    allok &= run_paged(lens, H, HKV, D, KM, c, pg)
print("=== RAGGED ===")
for lens, H, HKV, D, c, pg in configs:
    allok &= run_ragged(lens, H, HKV, D, KM, c)
# qo<kv append (q shorter than kv per request): causal aligns q to END of kv
print("=== qo<kv APPEND (causal end-aligned) ===")
for D in (128, 256):
    lens = [256, 400, 700]
    qol = [64, 100, 200]
    allok &= run_paged(lens, 8, 2, D, KM, True, 16, qo_lens=qol)
    allok &= run_ragged(lens, 8, 2, D, KM, True, qo_lens=qol)

# head_dim 64 must RAISE (guarded unsupported), not silently produce garbage.
print("=== head_dim 64 guard (must raise) ===")
guard_ok = True
try:
    run_ragged([96, 300, 700], 8, 2, 64, KM, True)
    print("  D64 int8 did NOT raise -> guard MISSING"); guard_ok = False
except (AssertionError, Exception) as ex:
    msg = str(ex)
    if "head_dim_vo==64" in msg or "head_dim" in msg:
        print(f"  D64 int8 correctly raised: {msg[:80]}")
    else:
        print(f"  D64 raised but unexpected msg: {msg[:120]}"); guard_ok = False
allok &= guard_ok
print("\nI5_SWEEP:", "ALL_PASS" if allok else "FAIL")
