#!/usr/bin/env python3
"""GATED + HALF-ONLY fp16-accum PV (v3) — flinfer 0.6.8 build-venv JIT.

File-scope template helpers make the gate template-DEPENDENT (proper if-constexpr discard) AND
half-only: FA_PV16<KTraits> = (FA_USE_FP16_PV != 0) && is_same_v<DTypeProb, half>. So bf16-prob
kernels (DTypeProb=__nv_bfloat16) auto-fall-back to stock fp32 (no corruption); only half-prob
(W4A8/int8-Q -> DTypeProb=half, or fp16-served) get the +27% uint32[4] o_acc path. No KernelTraits
edit (avoids version-specific struct anchors). Flip `#define FA_USE_FP16_PV` 1/0 to compile-test."""
import os, sys, shutil

F = os.environ.get("FA_PREFILL_CUH")
if not F:
    import flashinfer
    F = os.path.join(os.path.dirname(flashinfer.__file__),
                     "data/include/flashinfer/attention/prefill.cuh")
BAK = F + ".gatedpv0612_bak"


def restore():
    assert os.path.exists(BAK), f"no backup {BAK}"
    shutil.copy2(BAK, F); print("restored", F)


def apply():
    if not os.path.exists(BAK):
        shutil.copy2(F, BAK)
    s = open(BAK).read()

    def rep(o, n, c):
        k = s.count(o)
        assert k == c, f"anchor {k}!={c}: {o[:55]!r}"
        return s.replace(o, n)

    # 0. inject gate macro + half-only template helpers before init_states.
    helpers = (
        "#ifndef FA_USE_FP16_PV\n#define FA_USE_FP16_PV 1\n#endif\n"
        "template <typename T, typename = void> struct _fa_ptype { using type = void; };\n"
        "template <typename T> struct _fa_ptype<T, std::void_t<typename T::DTypeProb>> { using type = typename T::DTypeProb; };\n"
        "template <typename T, typename = void> struct _fa_qtype { using type = void; };\n"
        "template <typename T> struct _fa_qtype<T, std::void_t<typename T::DTypeQ>> { using type = typename T::DTypeQ; };\n"
        "template <typename KTraits>\n"
        "inline constexpr bool FA_PV16 = (FA_USE_FP16_PV != 0) && ("
        "std::is_same_v<typename _fa_ptype<KTraits>::type, half> || "
        "(std::is_void_v<typename _fa_ptype<KTraits>::type> && std::is_same_v<typename _fa_qtype<KTraits>::type, half>));\n"
        "template <typename KTraits>\n"
        "using FA_OFragT = std::conditional_t<FA_PV16<KTraits>, uint32_t, float>;\n"
        "template <typename KTraits>\n"
        "inline constexpr uint32_t FA_O_NREG = FA_PV16<KTraits> ? 4u : 8u;\n\n"
    )
    s = rep("template <typename KTraits>\n__device__ __forceinline__ void init_states(",
            helpers + "template <typename KTraits>\n__device__ __forceinline__ void init_states(", 1)

    # 1. loop-fn signatures
    s = rep("                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][8],\n",
            "                                            FA_OFragT<KTraits> (*o_frag)[KTraits::NUM_MMA_D_VO][FA_O_NREG<KTraits>],\n", 1)
    s = rep("    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],\n    float (*d)[2]) {\n",
            "    FA_OFragT<KTraits> (*o_frag)[KTraits::NUM_MMA_D_VO][FA_O_NREG<KTraits>], typename KTraits::DTypeQKAccum (*m)[2],\n    float (*d)[2]) {\n", 1)
    s = rep("    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], float (*d)[2]) {\n",
            "    FA_OFragT<KTraits> (*o_frag)[KTraits::NUM_MMA_D_VO][FA_O_NREG<KTraits>], float (*d)[2]) {\n", 1)

    # 2. init zero
    s = rep("      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {\n        o_frag[mma_q][mma_d][reg_id] = 0.f;\n      }",
            "      if constexpr (FA_PV16<KTraits>) {\n"
            "        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) o_frag[mma_q][mma_d][reg_id] = 0u;\n"
            "      } else {\n"
            "        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) o_frag[mma_q][mma_d][reg_id] = 0.f;\n"
            "      }", 1)

    # 3. rescale x2
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
        "#pragma unroll\n"
        "          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {\n"
        "            if constexpr (FA_PV16<KTraits>) {\n"
        "              half2 _os2 = __half2half2(__float2half(o_scale));\n"
        "              ((half2*)o_frag[mma_q][mma_d])[j] = __hmul2(((half2*)o_frag[mma_q][mma_d])[j], _os2);\n"
        "              ((half2*)o_frag[mma_q][mma_d])[j + 2] = __hmul2(((half2*)o_frag[mma_q][mma_d])[j + 2], _os2);\n"
        "            } else {\n"
        "              o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;\n"
        "              o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;\n"
        "              o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;\n"
        "              o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;\n"
        "            }\n"
        "          }", 2)

    # 4. PV MMA
    for sf in ("s_frag_f16", "s_frag"):
        s = rep(f"          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeProb>(\n"
                f"              o_frag[mma_q][mma_d], (uint32_t*){sf}[mma_q][mma_kv], b_frag);",
                f"          if constexpr (FA_PV16<KTraits>) {{\n"
                f"            mma::mma_sync_m16n16k16_row_col_f16f16f16(\n"
                f"                o_frag[mma_q][mma_d], (uint32_t*){sf}[mma_q][mma_kv], b_frag);\n"
                f"          }} else {{\n"
                f"            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeProb>(\n"
                f"                o_frag[mma_q][mma_d], (uint32_t*){sf}[mma_q][mma_kv], b_frag);\n"
                f"          }}", 1)

    # 5. body decl + calls
    s = rep("    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];\n",
            "    alignas(16) FA_OFragT<KTraits> o_acc[NUM_MMA_Q][NUM_MMA_D_VO][FA_O_NREG<KTraits>];\n", 3)
    s = rep("init_states<KTraits>(variant, o_frag, m, d);", "init_states<KTraits>(variant, o_acc, m, d);", 3)
    s = rep("update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);",
            "update_mdo_states<KTraits>(variant, s_frag, o_acc, m, d);", 3)
    s = rep(", s_frag, o_frag, d);",
            ", s_frag, o_acc, d);", 3)

    # 6. epilogue bridge
    s = rep("    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);",
            "    alignas(16) float _o_frag_mat[NUM_MMA_Q][NUM_MMA_D_VO][8];\n"
            "    float (*o_frag)[NUM_MMA_D_VO][8];\n"
            "    if constexpr (FA_PV16<KTraits>) {\n"
            "#pragma unroll\n"
            "      for (uint32_t _mq = 0; _mq < NUM_MMA_Q; ++_mq)\n"
            "#pragma unroll\n"
            "        for (uint32_t _md = 0; _md < NUM_MMA_D_VO; ++_md) {\n"
            "          float2 _f0 = __half22float2(((half2*)o_acc[_mq][_md])[0]);\n"
            "          float2 _f1 = __half22float2(((half2*)o_acc[_mq][_md])[1]);\n"
            "          float2 _f2 = __half22float2(((half2*)o_acc[_mq][_md])[2]);\n"
            "          float2 _f3 = __half22float2(((half2*)o_acc[_mq][_md])[3]);\n"
            "          _o_frag_mat[_mq][_md][0]=_f0.x; _o_frag_mat[_mq][_md][1]=_f0.y;\n"
            "          _o_frag_mat[_mq][_md][2]=_f1.x; _o_frag_mat[_mq][_md][3]=_f1.y;\n"
            "          _o_frag_mat[_mq][_md][4]=_f2.x; _o_frag_mat[_mq][_md][5]=_f2.y;\n"
            "          _o_frag_mat[_mq][_md][6]=_f3.x; _o_frag_mat[_mq][_md][7]=_f3.y;\n"
            "        }\n"
            "      o_frag = _o_frag_mat;\n"
            "    } else {\n"
            "      o_frag = reinterpret_cast<float (*)[NUM_MMA_D_VO][8]>(&o_acc[0][0][0]);\n"
            "    }\n"
            "    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);", 3)

    open(F, "w").write(s); print("patched GATED v3 (half-only) ->", F)


if __name__ == "__main__":
    restore() if "--restore" in sys.argv else apply()
