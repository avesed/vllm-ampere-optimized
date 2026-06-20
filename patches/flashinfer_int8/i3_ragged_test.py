#!/usr/bin/env python3
"""I-3 (ragged): int8-QK in BatchPrefillWithRaggedKVCacheWrapper (fa2), single request.
Confirms the 3rd compute_qk call site (ragged kernel) is wired + correct.
GATE: cos>0.99 AND |O_i8|/|O_ref| in [0.95,1.05]."""
import math, os, torch, torch.nn.functional as F
import flashinfer
torch.manual_seed(0)
dev = "cuda"
L = int(os.environ.get("L", "256")); H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", str(H))); D = int(os.environ.get("D", "128"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
dt = torch.float16; sm = 1.0 / math.sqrt(D)
q = torch.randn(L, H, D, device=dev, dtype=dt) * 0.5
k = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
v = torch.randn(L, HKV, D, device=dev, dtype=dt) * 0.5
qo_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=dev)
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wr.plan(qo_indptr, kv_indptr, H, HKV, D, causal=CAUSAL,
        q_data_type=dt, kv_data_type=dt, o_data_type=dt, pos_encoding_mode="NONE", sm_scale=sm)
O_ref = wr.run(q, k, v)

def qpt(x):
    s = torch.clamp(x.float().abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
    return torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8), s.to(torch.float32)
k_sm = (k.float() - k.float().mean(dim=0, keepdim=True)).to(dt)
q_i8, sq = qpt(q); k_i8, sk = qpt(k_sm)
sv = v.float().abs().amax() / 127.0
v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)

wi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
wi.plan(qo_indptr, kv_indptr, H, HKV, D, causal=CAUSAL,
        q_data_type=torch.int8, kv_data_type=torch.int8, o_data_type=torch.float16,
        pos_encoding_mode="NONE", sm_scale=sm)
O = wi.run(q_i8, k_i8, v_i8, sq, sk) * sv
a, b = O.flatten().float(), O_ref.flatten().float()
fin = torch.isfinite(a) & torch.isfinite(b)
cos = F.cosine_similarity(a[fin], b[fin], dim=0).item()
ratio = (a[fin].norm() / b[fin].norm()).item()
print(f"L={L} H={H} HKV={HKV} D={D} causal={CAUSAL}")
print(f"=== RAGGED cos = {cos:.5f}  |O_i8|/|O_ref| = {ratio:.4f} ===")
ok = (cos > 0.99) and (0.95 <= ratio <= 1.05) and not bool(torch.isnan(O).any())
print("RESULT:", "RAGGED_PASS" if ok else "FAIL")
