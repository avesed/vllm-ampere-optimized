#!/usr/bin/env python3
"""I-3 paged sweep: int8-QK BatchPrefillWithPagedKVCacheWrapper vs fp16 paged ref.
PASS = cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05]. Includes non-page-aligned L."""
import math, torch, torch.nn.functional as F
import flashinfer
torch.manual_seed(0)
dev = "cuda"
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

def run(L, H, HKV, D, PAGE, causal):
    dt = torch.float16
    sm = 1.0 / math.sqrt(D)
    q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
    k = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
    num_pages = (L + PAGE - 1) // PAGE
    last = L - (num_pages - 1) * PAGE
    def to_paged(x):
        pad = num_pages * PAGE - L
        if pad:
            x = torch.cat([x, torch.zeros(pad, HKV, D, device=dev, dtype=x.dtype)], 0)
        return x.view(num_pages, PAGE, HKV, D).contiguous()
    qo_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=dev)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=dev)
    kv_last = torch.tensor([last], dtype=torch.int32, device=dev)

    wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wr.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, PAGE,
            causal=causal, q_data_type=dt, kv_data_type=dt, o_data_type=dt,
            pos_encoding_mode="NONE", sm_scale=sm)
    O_ref = wr.run(q, (to_paged(k), to_paged(v)))

    def qpt(x):
        s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
        xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
        return xi, s.to(torch.float32)
    k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
    q_i8, sq = qpt(q); k_i8, sk = qpt(k_sm)
    sv = v.float().abs().amax() / 127.0
    v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
    wi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wi.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, PAGE,
            causal=causal, q_data_type=torch.int8, kv_data_type=torch.int8,
            o_data_type=torch.float16, pos_encoding_mode="NONE", sm_scale=sm)
    O = wi.run(q_i8, (to_paged(k_i8), to_paged(v_i8)), sq, sk) * sv
    a, b = O.flatten().float(), O_ref.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
    ratio = (a[fin].norm() / b[fin].norm()).item()
    return cos, ratio, bool(torch.isnan(O).any())

rows = []
cases = [
    (256, 8, 8, 128, 16, True), (256, 8, 2, 128, 16, True),
    (256, 8, 8, 128, 16, False), (200, 8, 8, 128, 16, True),   # non-page-aligned
    (2048, 8, 8, 128, 16, True), (2048, 8, 2, 128, 16, True),
    (256, 8, 8, 256, 16, True), (256, 8, 2, 256, 16, True),
    (2048, 8, 8, 256, 16, True), (2048, 8, 2, 256, 32, False),
    (333, 8, 2, 256, 16, True),   # non-aligned + GQA + D256
]
for (L, H, HKV, D, PAGE, causal) in cases:
    cos, ratio, nan = run(L, H, HKV, D, PAGE, causal)
    ok = (cos > 0.99) and (0.95 <= ratio <= 1.05) and not nan
    rows.append(ok)
    print(f"L={L:5d} H={H} HKV={HKV} D={D} page={PAGE} causal={int(causal)} | "
          f"cos={cos:.5f} mag={ratio:.4f} {'PASS' if ok else 'FAIL'}")
print("\nPAGED_SWEEP:", "ALL_PASS" if all(rows) else f"{sum(rows)}/{len(rows)} pass")
