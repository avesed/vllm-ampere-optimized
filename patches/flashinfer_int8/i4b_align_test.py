#!/usr/bin/env python3
"""I-4b chunked-prefix CORRECTNESS isolation test: verify single_prefill_with_kv_cache
with qo_len < kv_len + causal=True aligns the NEW chunk's queries to the END of the kv
(query token i attends to kv[0 .. kv_len-qo_len+i]) — the exact semantics chunked prefill
needs (the new chunk's tokens attend causally to all prior context + themselves).

We build a 2-chunk scenario:
  * full prompt of length S (= prefix P + new chunk Q_new of length C, S = P + C)
  * REFERENCE = full single_prefill over the whole prompt (causal), take the LAST C rows
    of the output = the rows for the new chunk's tokens.
  * CANDIDATE = single_prefill(q = new chunk's queries [C], k/v = FULL context [S],
    causal=True). If FlashInfer aligns qo to the END of kv, candidate == reference.

Run for BOTH fp16 (proves the alignment convention) AND int8 (proves the int8 kernel
preserves it). PASS gate: cos > 0.999 between candidate and the last-C reference rows,
for both dtypes, over several (P, C) splits incl page-unaligned.

Env: D (head_dim, default 256), H (default 8), HKV (default 2 -> GQA g4).
"""
import math, os, torch, torch.nn.functional as F
import flashinfer

torch.manual_seed(0)
dev = "cuda"
D = int(os.environ.get("D", "256"))
H = int(os.environ.get("H", "8"))
HKV = int(os.environ.get("HKV", "2"))
dtype = torch.float16
sm = 1.0 / math.sqrt(D)


def q_pertoken(x):  # x:[L,Hx,D] -> int8, scale[L]
    s = x.float().abs().amax(dim=(1, 2)) / 127.0
    s = torch.clamp(s, min=1e-8)
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)


def run_fp16(q, k, v, causal):
    return flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa2", pos_encoding_mode="NONE", sm_scale=sm)


def run_int8(q, k, v, causal):
    # smooth_k over the FULL kv (per (head,channel) mean) — matches backend
    k_mean = k.float().mean(dim=0, keepdim=True)
    k_sm = (k.float() - k_mean).to(k.dtype)
    q_i8, sq = q_pertoken(q)
    k_i8, sk = q_pertoken(k_sm)
    sv = float(torch.clamp(v.float().abs().amax() / 127.0, min=1e-8))
    v_i8 = torch.clamp(torch.round(v.float() / sv), -127, 127).to(torch.int8)
    o = flashinfer.single_prefill_with_kv_cache(
        q_i8, k_i8, v_i8, scale_q=sq, scale_k=sk, causal=causal,
        backend="fa2", o_dtype=torch.float16, pos_encoding_mode="NONE", sm_scale=sm)
    return o * sv


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    return F.cosine_similarity(a[fin], b[fin], dim=0).item()


def test_split(P, C, fn, label):
    S = P + C
    q_full = torch.randn(S, H, D, device=dev, dtype=dtype) * 0.5
    k_full = torch.randn(S, HKV, D, device=dev, dtype=dtype) * 0.5
    v_full = torch.randn(S, HKV, D, device=dev, dtype=dtype) * 0.5

    # REFERENCE: full causal prefill, take last C rows (= the new chunk's output)
    o_full = fn(q_full, k_full, v_full, causal=True)
    o_ref_newchunk = o_full[P:S]

    # CANDIDATE: new chunk's queries attending to FULL context, causal=True (qo<kv)
    q_new = q_full[P:S]                  # [C,H,D]
    o_cand = fn(q_new, k_full, v_full, causal=True)   # qo=C, kv=S

    c = cos(o_cand, o_ref_newchunk)
    ok = c > 0.999
    print(f"  [{label}] P={P} C={C} S={S}  cos(cand, ref_last{C})={c:.5f}  {'OK' if ok else 'FAIL'}")
    return ok


print(f"=== chunked-prefix qo<kv causal-alignment test  D={D} H={H} HKV={HKV} ===")
splits = [(1024, 512), (2000, 333), (4096, 1024), (100, 28), (777, 256), (8192, 2048)]
all_ok = True
print("-- fp16 (alignment convention) --")
for P, C in splits:
    all_ok &= test_split(P, C, run_fp16, "fp16")
print("-- int8 (kernel preserves alignment) --")
for P, C in splits:
    all_ok &= test_split(P, C, run_int8, "int8")

# Sanity NEGATIVE control: if we (wrongly) treated qo as aligned to the START of kv,
# the candidate would NOT match the last-C rows. Verify by comparing cand to the FIRST C rows.
print("-- negative control (cand vs FIRST C rows of full output; expect LOW cos) --")
P, C = 4096, 1024
S = P + C
q_full = torch.randn(S, H, D, device=dev, dtype=dtype) * 0.5
k_full = torch.randn(S, HKV, D, device=dev, dtype=dtype) * 0.5
v_full = torch.randn(S, HKV, D, device=dev, dtype=dtype) * 0.5
o_full = run_fp16(q_full, k_full, v_full, causal=True)
o_cand = run_fp16(q_full[P:S], k_full, v_full, causal=True)
c_first = cos(o_cand, o_full[0:C])
c_last = cos(o_cand, o_full[P:S])
print(f"  cand vs first{C}={c_first:.5f}  cand vs last{C}={c_last:.5f}  "
      f"(end-aligned confirmed if last>>first)")

print("\nRESULT:", "ALIGN_PASS" if all_ok else "ALIGN_FAIL")
