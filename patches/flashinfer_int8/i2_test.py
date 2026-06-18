#!/usr/bin/env python3
"""I-2 validation: int8-QK FA2 single_prefill output vs fp16 reference (cosine similarity).

Strategy (zero new plumbing): per-TENSOR symmetric int8 quant of Q,K (+ smooth_k per-channel
mean subtract on K, absorbed by softmax shift-invariance), V per-tensor int8 (uniform V-scale
error is cosine-invariant). Fold sq*sk into sm_scale so the kernel's raw s32 logits become
Q_f16@K_f16^T/sqrt(d). compute_qk does s32->float only. PV stays fp16 (existing upcast).

Target: cos>0.95 = I-2 done; cos>0.90 = QK layout correct; cos~0 = layout wrong.
"""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
L = int(os.environ.get("L", "256"))
H = int(os.environ.get("H", "8"))
D = 128
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
dtype = torch.float16

q = torch.randn(L, H, D, device=dev, dtype=dtype) * 0.5
k = torch.randn(L, H, D, device=dev, dtype=dtype) * 0.5
v = torch.randn(L, H, D, device=dev, dtype=dtype) * 0.5
sm = 1.0 / math.sqrt(D)

# ---- fp16 reference ----
O_ref = flashinfer.single_prefill_with_kv_cache(
    q, k, v, causal=CAUSAL, backend="fa2", pos_encoding_mode="NONE", sm_scale=sm)
print("O_ref", tuple(O_ref.shape), O_ref.dtype, "nan?", bool(torch.isnan(O_ref).any()))

# ---- per-tensor symmetric int8 quant (smooth_k: subtract per-channel mean of K) ----
def q_pertensor(x):
    s = x.abs().amax() / 127.0
    return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8), float(s)

k_mean = k.float().mean(dim=0, keepdim=True)        # [1,H,D] per-(head,channel) mean
k_sm = (k.float() - k_mean).to(dtype)               # smooth_k
q_i8, sq = q_pertensor(q.float())
k_i8, sk = q_pertensor(k_sm)
v_i8, sv = q_pertensor(v.float())
print(f"scales  sq={sq:.5f} sk={sk:.5f} sv={sv:.5f}")

# fold sq*sk into sm_scale; smooth_k mean term is per-row-constant -> softmax-invariant.
# INT8_QK_DIV compensates the in-kernel INT8_QK_RCP=1/256 magnitude normalization (keeps the
# softmax exact while making FlashInfer's finite mask-fill (-5e4) dominate the causal mask).
INT8_QK_DIV = 256.0
sm_int8 = sm * sq * sk * INT8_QK_DIV

O_i8 = flashinfer.single_prefill_with_kv_cache(
    q_i8, k_i8, v_i8, causal=CAUSAL, backend="fa2", o_dtype=torch.float16,
    pos_encoding_mode="NONE", sm_scale=sm_int8)
print("O_i8 ", tuple(O_i8.shape), O_i8.dtype, "nan?", bool(torch.isnan(O_i8).any()),
      "inf?", bool(torch.isinf(O_i8).any()))

a = O_i8.flatten().float()
b = O_ref.flatten().float()
fin = torch.isfinite(a) & torch.isfinite(b)
print("finite frac:", fin.float().mean().item())
if fin.any():
    aa, bb = a[fin], b[fin]
    cos = F.cosine_similarity(aa, bb, dim=0).item()
    ratio = (aa.norm() / bb.norm()).item()
else:
    cos, ratio = float("nan"), float("nan")
print(f"\n=== cos = {cos:.5f}  (|O_i8|/|O_ref| = {ratio:.4f}) ===")
print("RESULT:", "I2_DONE" if cos > 0.95 else ("LAYOUT_OK" if cos > 0.90 else "LAYOUT_WRONG"))
