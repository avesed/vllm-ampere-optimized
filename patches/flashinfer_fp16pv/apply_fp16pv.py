#!/usr/bin/env python3
"""fp16-accumulate PV (pure-fp16 o_frag) for FlashInfer's prefill kernel — EXPERIMENTAL, sm_86 GeForce only.

WHAT IT DOES
  Replaces the prefill PV accumulator `o_frag` (fp32, 8 regs/fragment) with a pure-fp16 loop
  accumulator `o_acc` (uint32[4] half-packed, 4 regs) that accumulates P@V via the f16f16f16 HMMA
  (mma.cuh:629) with fp16 rescale, then materializes back into the kept float `o_frag[8]` after the
  kv loop (so threadblock_sync / transform_output / write_o_reg_gmem are untouched). On GA10x consumer
  Ampere (sm_86: RTX 3090/3080/3070) the f16-input/f16-accum HMMA runs at 2x the f16/f32-accum rate,
  AND halving o_frag (128->64 regs at hd256) eliminates the 255-reg-cap stack spill — the spill relief,
  not the 2x-MMA, is the dominant effect.

  Fragment layout (verified mma.cuh:590): the m16n16k16 f16f16f16 output is C_h[0..3] (uint32 half2),
  C_h[i] = half2(C_f32[2i], C_f32[2i+1]); row j -> C_h[j], C_h[j+2]. The rescale multiplies those two
  half2 by o_scale; the materialize unpacks C_h[i] -> o_frag[2i], o_frag[2i+1].

MEASURED (sandbox RTX 3090 sm_86, flashinfer 0.6.8.post1 JIT, hd256 f16 single_prefill)
  - register: spilling kernels 9->0 (split probe) / 9->4 (this materialize variant); STACK 96->0/80.
  - op-level: +25.6% / +25.2% / +23.7% at 2k / 4k / 8k prefill; worst-row cos 0.999995/0.999990/0.999979.
  - accuracy (fake-quant cos vs fp32, sweep tile x ctx x flatness incl uniform-worst, validate_accuracy.py):
    pure-fp16 worst-row cos 0.999928 (~= shipped int8-QK 0.99992); two-level 0.999999.
  - e2e (vllm bench serve, 9B-w4a16 single-card, attention_backend=FLASHINFER + dtype=fp16):
    prefill TTFT +1.1% @16k / +2.1% @32k / +4.1% @64k (grows with ctx; hybrid 8/32 full-attn dilution).

STATUS / LIMITS — NOT wired into the default build; DO NOT default-on.
  - GeForce-GA10x sm_86 ONLY. A100 (sm_80) and pro sm_86 (A40/A6000/A10) run f32-accum at FULL rate ->
    ZERO benefit AND fp16-accum is LESS precise there. A `__CUDA_ARCH__>=860` gate is WRONG; a runtime
    GeForce-SKU / rate probe is REQUIRED before default-on and is UNBUILT.
  - DTypeProb=half only (W4A8/int8-Q or fp16-served). bf16-compute needs a P->fp16 / V bf16->fp16 cast
    (not applied here) — the f16f16f16 primitive is half-only.
  - prefill-only (decode is GEMV bandwidth-bound; 0 gain). On tp2 ~0 e2e (TP all-reduce ~67% of prefill).
  - This patch is UNCONDITIONAL (always fp16 o_frag) — for the bench. Productionizing for default-on needs
    a USE_FP16_PV_REDUCTION template flag + JIT URI key (gated off) + the SKU probe + an autoregressive
    accuracy gate on real W4A8 (current accuracy is fake-quant op-level, NOT closed e2e). See NOTES.md.

USAGE (applies to the INSTALLED-PACKAGE layout, like patches/flashinfer_int8/*.py):
  python3 apply_fp16pv.py            # patch (backs up prefill.cuh.fp16pv_bak)
  python3 apply_fp16pv.py --restore  # restore the backup
Anchors are strict-asserted: they FAIL LOUDLY on version skew (proof the edit still matches the source).
"""
import os, sys, shutil

import flashinfer  # noqa: E402
F = os.path.join(os.path.dirname(flashinfer.__file__),
                 "data/include/flashinfer/attention/prefill.cuh")
BAK = F + ".fp16pv_bak"


def restore():
    assert os.path.exists(BAK), f"no backup {BAK}"
    shutil.copy2(BAK, F)
    print("restored", F)


def apply():
    if not os.path.exists(BAK):
        shutil.copy2(F, BAK)
    s = open(F).read()

    def rep(o, n, c):
        k = s.count(o)
        assert k == c, f"anchor count {k}!={c} (version skew?): {o[:60]!r}"
        return s.replace(o, n)

    # 1. signatures float[8]->uint32_t[4] on init/update/compute only (verify/transform/write stay fp32[8])
    s = rep("__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,\n"
            "                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][8],\n",
            "__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,\n"
            "                                            uint32_t (*o_frag)[KTraits::NUM_MMA_D_VO][4],\n", 1)
    s = rep("    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],\n    float (*d)[2]) {\n",
            "    uint32_t (*o_frag)[KTraits::NUM_MMA_D_VO][4], typename KTraits::DTypeQKAccum (*m)[2],\n    float (*d)[2]) {\n", 1)
    s = rep("    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], float (*d)[2]) {\n",
            "    uint32_t (*o_frag)[KTraits::NUM_MMA_D_VO][4], float (*d)[2]) {\n", 1)

    # 2. init: zero 4 uint32
    s = rep("      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {\n        o_frag[mma_q][mma_d][reg_id] = 0.f;",
            "      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {\n        o_frag[mma_q][mma_d][reg_id] = 0u;", 1)

    # 3. online-softmax rescale -> half2 (both DTypeQKAccum branches share this block)
    s = rep(
        "          d[mma_q][j] *= o_scale;\n"
        "#pragma unroll\n"
        "          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {\n"
        "            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;\n"
        "            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;\n"
        "            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;\n"
        "            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;\n"
        "          }",
        "          d[mma_q][j] *= o_scale;\n"
        "          half2 _os2 = __half2half2(__float2half(o_scale));\n"
        "#pragma unroll\n"
        "          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {\n"
        "            ((half2*)o_frag[mma_q][mma_d])[j] = __hmul2(((half2*)o_frag[mma_q][mma_d])[j], _os2);\n"
        "            ((half2*)o_frag[mma_q][mma_d])[j + 2] = __hmul2(((half2*)o_frag[mma_q][mma_d])[j + 2], _os2);\n"
        "          }", 2)

    # 4. PV mma f16f16f32 -> f16f16f16 into o_acc (both branches)
    for sf in ("s_frag_f16", "s_frag"):
        s = rep(f"          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(\n"
                f"              o_frag[mma_q][mma_d], (uint32_t*){sf}[mma_q][mma_kv], b_frag);",
                f"          mma::mma_sync_m16n16k16_row_col_f16f16f16(\n"
                f"              o_frag[mma_q][mma_d], (uint32_t*){sf}[mma_q][mma_kv], b_frag);", 1)

    # 5. kernel bodies: o_acc uint32[4] loop accumulator, route loop fns to it, materialize half->float
    s = rep("    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];\n",
            "    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];\n"
            "    alignas(16) uint32_t o_acc[NUM_MMA_Q][NUM_MMA_D_VO][4];\n", 3)
    s = rep("init_states<KTraits>(variant, o_frag, m, d);", "init_states<KTraits>(variant, o_acc, m, d);", 3)
    s = rep("update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);",
            "update_mdo_states<KTraits>(variant, s_frag, o_acc, m, d);", 3)
    s = rep("compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);",
            "compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_acc, d);", 3)
    s = rep("    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);",
            "#pragma unroll\n"
            "    for (uint32_t _mq = 0; _mq < NUM_MMA_Q; ++_mq)\n"
            "#pragma unroll\n"
            "      for (uint32_t _md = 0; _md < NUM_MMA_D_VO; ++_md) {\n"
            "        float2 _f0 = __half22float2(((half2*)o_acc[_mq][_md])[0]);\n"
            "        float2 _f1 = __half22float2(((half2*)o_acc[_mq][_md])[1]);\n"
            "        float2 _f2 = __half22float2(((half2*)o_acc[_mq][_md])[2]);\n"
            "        float2 _f3 = __half22float2(((half2*)o_acc[_mq][_md])[3]);\n"
            "        o_frag[_mq][_md][0]=_f0.x; o_frag[_mq][_md][1]=_f0.y;\n"
            "        o_frag[_mq][_md][2]=_f1.x; o_frag[_mq][_md][3]=_f1.y;\n"
            "        o_frag[_mq][_md][4]=_f2.x; o_frag[_mq][_md][5]=_f2.y;\n"
            "        o_frag[_mq][_md][6]=_f3.x; o_frag[_mq][_md][7]=_f3.y;\n"
            "      }\n"
            "    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);", 3)

    open(F, "w").write(s)
    print("patched fp16-PV ->", F, "(backup:", BAK + ")")


if __name__ == "__main__":
    restore() if "--restore" in sys.argv else apply()
