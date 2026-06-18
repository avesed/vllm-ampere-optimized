#!/usr/bin/env python3
"""I-3 validation: int8-QK in the PAGED kernel (BatchPrefillWithPagedKVCacheWrapper, fa2).

Single request, paged KV cache. compute_qk is SHARED with single/ragged so the int8 matmul is
reused; this exercises the SEPARATE paged kernel instantiation (its compute_sfm_v / page_produce_kv
loads + the DTypeProb/load_q fixes + the per-token dequant token-index math adapted to paged kv).

Reference = fp16 PAGED wrapper (same paged path) so we isolate the int8 effect.
Int8 = per-token symmetric int8 quant (smooth_k on K), scale_q/scale_k passed positionally
(run(q_i8, (k_i8,v_i8), sq, sk)), sm_scale = 1/sqrt(d). V dequant applied to output.

GATE: cos > 0.99 AND |O_i8|/|O_ref| in [0.95, 1.05].
"""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
L = int(os.environ.get("L", "256"))          # kv_len == qo_len (full prefill, single request)
H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", str(H)))
D = int(os.environ.get("D", "128"))
PAGE = int(os.environ.get("PAGE", "16"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
dt = torch.float16
sm = 1.0 / math.sqrt(D)

q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
k = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5

num_pages = (L + PAGE - 1) // PAGE
last_page_len = L - (num_pages - 1) * PAGE

# ---- paged layout helpers (NHD: [num_pages, page_size, num_kv_heads, head_dim]) ----
def to_paged(x):  # x: [L, HKV, D] -> [num_pages, PAGE, HKV, D]
    pad = num_pages * PAGE - L
    if pad:
        x = torch.cat([x, torch.zeros(pad, HKV, D, device=dev, dtype=x.dtype)], 0)
    return x.view(num_pages, PAGE, HKV, D).contiguous()

qo_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=dev)
kv_indices = torch.arange(num_pages, dtype=torch.int32, device=dev)
kv_last = torch.tensor([last_page_len], dtype=torch.int32, device=dev)

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

# ---- fp16 paged reference ----
wr_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wr_ref.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, PAGE,
            causal=CAUSAL, q_data_type=dt, kv_data_type=dt, o_data_type=dt,
            pos_encoding_mode="NONE", sm_scale=sm)
O_ref = wr_ref.run(q, (to_paged(k), to_paged(v)))

# ---- per-token int8 quant (smooth_k on K) ----
def qpt(x):
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)

k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
q_i8, sq = qpt(q)            # sq: [L] q tokens
k_i8, sk = qpt(k_sm)         # sk: [L] kv tokens (logical order)
sv = v.float().abs().amax() / 127.0
v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)

print(f"L={L} H={H} HKV={HKV} D={D} page={PAGE} pages={num_pages} last={last_page_len} causal={CAUSAL}")

wr_i8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wr_i8.plan(qo_indptr, kv_indptr, kv_indices, kv_last, H, HKV, D, PAGE,
           causal=CAUSAL, q_data_type=torch.int8, kv_data_type=torch.int8,
           o_data_type=torch.float16, pos_encoding_mode="NONE", sm_scale=sm)
# pass per-token scale_q/scale_k positionally (extracted into fp8_scale_q/k for int8 q)
O_i8 = wr_i8.run(q_i8, (to_paged(k_i8), to_paged(v_i8)), sq, sk) * sv

a, b = O_i8.flatten().float(), O_ref.flatten().float()
fin = torch.isfinite(a) & torch.isfinite(b)
nan = bool(torch.isnan(O_i8).any())
if fin.any():
    cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
    ratio = (a[fin].norm() / b[fin].norm()).item()
else:
    cos, ratio = float("nan"), float("nan")
print(f"finite={fin.float().mean().item():.4f} nan={nan}")
print(f"\n=== PAGED cos = {cos:.5f}   |O_i8|/|O_ref| = {ratio:.4f} ===")
ok = (cos > 0.99) and (0.95 <= ratio <= 1.05) and not nan
print("RESULT:", "I3_PASS" if ok else ("COS_OK_MAG_BAD" if cos > 0.99 else "FAIL"))
