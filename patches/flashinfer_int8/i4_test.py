#!/usr/bin/env python3
"""I-4a validation: PRODUCTION int8-QK single_prefill with REAL per-token dequant.

Removes the I-2 test hack entirely:
  * REAL per-token symmetric int8 quant: sq[token] = max|q[token]|/127 (per q token),
    sk[token] = max|k_smooth[token]|/127 (per kv token), smooth_k = subtract per-(head,channel)
    mean of K (absorbed by softmax row-shift invariance).
  * Pass scale_q=sq, scale_k=sk as tensors; sm_scale = 1/sqrt(d) (NO sq*sk folding, NO ×256).
  * The kernel applies sf = acc * sq[q] * sk[kv] in-kernel (real dequant).

GATE (the proof real dequant works): cos > 0.99 AND |O_i8|/|O_ref| in [0.95, 1.05]
(magnitude now CORRECT vs the hack's 53×). Sweeps L, D, causal/non-causal, GQA via env.
"""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
L = int(os.environ.get("L", "256"))
H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", str(H)))   # GQA when HKV < H
D = int(os.environ.get("D", "128"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
dtype = torch.float16

q = torch.randn(L, H, D, device=dev, dtype=dtype) * 0.5
k = torch.randn(L, HKV, D, device=dev, dtype=dtype) * 0.5
v = torch.randn(L, HKV, D, device=dev, dtype=dtype) * 0.5
sm = 1.0 / math.sqrt(D)

# ---- fp16 reference ----
O_ref = flashinfer.single_prefill_with_kv_cache(
    q, k, v, causal=CAUSAL, backend="fa2", pos_encoding_mode="NONE", sm_scale=sm)

# ---- REAL per-token symmetric int8 quant ----
def q_pertoken(x):  # x: [L, Hx, D] -> int8 [L,Hx,D], scale [L] (per token, across heads*dim)
    # per-token scalar over (head,dim) so one scale per token row (target = [num_tokens])
    s = x.float().abs().amax(dim=(1, 2)) / 127.0          # [L]
    s = torch.clamp(s, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)

# smooth_k: subtract per-(head,channel) mean of K (per-row-constant logit shift -> softmax-invariant)
k_mean = k.float().mean(dim=0, keepdim=True)              # [1, HKV, D]
k_sm = (k.float() - k_mean).to(dtype)

q_i8, sq = q_pertoken(q)                                  # sq: [L] (q tokens)
k_i8, sk = q_pertoken(k_sm)                               # sk: [L] (kv tokens)
# V per-tensor int8 (uniform V-scale error is cosine-invariant; here we make magnitude exact)
sv = v.float().abs().amax() / 127.0
v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)

print(f"L={L} H={H} HKV={HKV} D={D} causal={CAUSAL}  sq[:3]={sq[:3].tolist()} sk[:3]={sk[:3].tolist()} sv={sv:.5f}")

# REAL dequant: sm_scale stays 1/sqrt(d); pass per-token scale tensors.
O_i8 = flashinfer.single_prefill_with_kv_cache(
    q_i8, k_i8, v_i8, scale_q=sq, scale_k=sk,
    causal=CAUSAL, backend="fa2", o_dtype=torch.float16,
    pos_encoding_mode="NONE", sm_scale=sm)
# V dequant (out *= sv) to restore output magnitude
O_i8 = O_i8 * sv

a = O_i8.flatten().float()
b = O_ref.flatten().float()
fin = torch.isfinite(a) & torch.isfinite(b)
nan_i8 = bool(torch.isnan(O_i8).any())
if fin.any():
    aa, bb = a[fin], b[fin]
    cos = F.cosine_similarity(aa, bb, dim=0).item()
    ratio = (aa.norm() / bb.norm()).item()
else:
    cos, ratio = float("nan"), float("nan")
print(f"finite={fin.float().mean().item():.4f} nan_i8={nan_i8}")
print(f"\n=== cos = {cos:.5f}   |O_i8|/|O_ref| = {ratio:.4f} ===")
mag_ok = 0.95 <= ratio <= 1.05
cos_ok = cos > 0.99
print("RESULT:", "I4A_PASS" if (cos_ok and mag_ok) else
      ("COS_OK_MAG_BAD" if cos_ok else ("MAG_OK_COS_BAD" if mag_ok else "FAIL")))
