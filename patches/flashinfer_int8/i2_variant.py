#!/usr/bin/env python3
"""Perturbation-profiling variants of the I-2 int8 compute_qk (ncu unavailable in image).
Applies i2_compute_qk.py (full int8 branch) then, per VARIANT env, neuters one component to
localize the ~1.1x ceiling:
  full   - real direct-index smem load + real s8s8s32 IMMA (baseline)
  nomma  - real load, IMMA replaced by cheap XOR-add ALU (A,B still consumed -> loads survive)
           -> isolates IMMA cost. nomma~=full => kernel NOT tensor-core bound.
  noload - smem reads replaced by an arithmetic hash of (row,dcol) (zero smem traffic), real IMMA
           -> isolates direct-index load cost. noload~=full => load NOT the bottleneck;
              noload<<full => load IS the bottleneck (tunable/recoverable).
Each variant needs its own JIT build (rm cache); run in a fresh --rm container from pristine pkg.
"""
import os, re, flashinfer
VARIANT = os.environ.get("VARIANT", "full")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "i2_compute_qk.py")).read())  # writes the full int8 branch into P

FI = os.path.dirname(flashinfer.__file__)
P = os.path.join(FI, "data/include/flashinfer/attention/prefill.cuh")
s = open(P).read()

if VARIANT == "nomma":
    s2 = re.sub(r'asm volatile\(\s*"mma\.sync\.aligned\.m16n8k32.*?"r"\(c\[3\]\)\);',
                'c[0]+=(int32_t)(A[0]^B[0]);c[1]+=(int32_t)(A[1]^B[1]);'
                'c[2]+=(int32_t)(A[2]^B[0]);c[3]+=(int32_t)(A[3]^B[1]);',
                s, count=1, flags=re.DOTALL)
    assert s2 != s, "nomma: asm regex no match"
    s = s2
elif VARIANT == "noload":
    for nm in ("ldq", "ldk"):
        s2 = re.sub(r'auto ' + nm + r' = \[&\]\(uint32_t row, uint32_t dcol\) -> uint32_t \{.*?\};',
                    'auto ' + nm + ' = [&](uint32_t row, uint32_t dcol) -> uint32_t { '
                    '(void)q_base;(void)k_base;(void)US_Q;(void)US_K; '
                    'return row*2654435761u + dcol*40503u + 0x9e3779b9u; };',
                    s, count=1, flags=re.DOTALL)
        assert s2 != s, nm + ": lambda regex no match"
        s = s2
elif VARIANT != "full":
    raise SystemExit("unknown VARIANT " + VARIANT)

open(P, "w").write(s)
print("VARIANT_APPLIED", VARIANT)
