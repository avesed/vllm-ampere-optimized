#!/usr/bin/env python3
"""I-1 (DONE, validated): admit int8 q/kv through FlashInfer 0.6.12 prefill JIT so an int8
single_prefill COMPILES + is callable on sm_80/86. Numerics are PLACEHOLDER (int8 rides the
existing f16-upcast compute_qk path -> garbage); I-2 wires the real int8 IMMA into compute_qk.

Dev rig: run inside the v0.23.0 image; mutates the installed flashinfer package in-place
(back up to *.orig once). Then `rm -rf $HOME/.cache/flashinfer` and build:
    get_single_prefill_module("fa2", torch.int8, torch.int8, torch.float16, 128,128,0,False,False,False)
GATE PASS = module builds (no KeyError / static_assert / fp8-assert). Verified 2026-06-18.

Ship as real patches/*.patch later; this script is the reproducible edit set.
"""
import os, flashinfer
FI = os.path.dirname(flashinfer.__file__)

def patch(rel, subs, required=True):
    p = os.path.join(FI, rel); s = open(p).read(); orig = s; n = 0
    for a, b in subs:
        c = s.count(a); s = s.replace(a, b); n += c
    if s != orig:
        if not os.path.exists(p + ".orig"): open(p + ".orig", "w").write(orig)
        open(p, "w").write(s)
    print(f"  {rel}: {n} edit(s)")
    if required and n == 0: raise SystemExit(f"FAILED: no edit applied in {rel} (version drift?)")

# E1 — dtype_map_kv lacks int8 (KeyError blocker); dtype_map + filename_safe already have it.
patch("jit/utils.py", [('dtype_map_kv = {\n', 'dtype_map_kv = {\n    torch.int8: "int8_t",\n')])

# E4/E5 — 3x kernel-entry guard on int8 Q (sizeof 1).
patch("data/include/flashinfer/attention/prefill.cuh",
      [("static_assert(sizeof(DTypeQ) == 2);", "static_assert(sizeof(DTypeQ) == 2 || sizeof(DTypeQ) == 1);")])

# E6a — enable define (sm80+, NOT the >=890 fp8 block). E_rowsum — relax the f16 rowsum guard
# that the placeholder path trips (m16k16_rowsum_f16f16f32<DTypeQ=int8_t>).
patch("data/include/flashinfer/mma.cuh", [
    ("#define FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED\n",
     "#define FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED\n#define FLASHINFER_MMA_S8S8S32_M16N8K32_ENABLED\n"),
    ('static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");',
     'static_assert(sizeof(DType) == 2 || sizeof(DType) == 1, "relaxed int8 placeholder I-1");'),
])

# E6b — paste the int8 IMMA wrapper (mma_s8s8s32_wrapper.cuh) after the f8f8f32 wrapper (brace-match).
WRAPPER = open(os.path.join(os.path.dirname(__file__), "mma_s8s8s32_wrapper.cuh")).read()
WRAPPER = WRAPPER[WRAPPER.index("template <typename T"):]  # drop the leading comment block
p = os.path.join(FI, "data/include/flashinfer/mma.cuh"); s = open(p).read()
if "mma_sync_m16n16k32_row_col_s8s8s32" not in s:
    i = s.index("void mma_sync_m16n16k32_row_col_f8f8f32"); b = s.index("{", i); depth = 0; j = b
    while j < len(s):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0: break
        j += 1
    open(p, "w").write(s[:j + 1] + "\n\n" + WRAPPER + s[j + 1:])
    print("  mma.cuh: int8 wrapper inserted")
print("I1_APPLIED")
