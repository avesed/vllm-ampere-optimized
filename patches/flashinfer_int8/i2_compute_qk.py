#!/usr/bin/env python3
"""I-2: wire the validated int8 IMMA (m16n8k32 s8s8s32) into FlashInfer FA2 compute_qk.

Applies ON TOP of i1_apply.py (run i1 first). Edits the installed prefill.cuh in-place:
  (A) add `const dim3 tid = threadIdx` to compute_qk's signature so the int8 branch can
      recover per-warp tile bases (get_warp_idx_q/kv).
  (B) inject an `if constexpr (sizeof(DTypeQ)==1)` branch at the top of compute_qk that:
        - directly indexes the b128-swizzled int8 Q/K smem (BYPASS ldmatrix) to build the
          m16n8k32 A(4xu32)/B(2xu32 per n8) fragments in the VALIDATED layout
          (g=lane>>2, t=lane&3; A=Q[g|g+8][base+t*4+{0..3|16..19}]; B=K[n][base+...]),
        - runs mma.m16n8k32.row.col.s32.s8.s8.s32 into an int32 accumulator,
        - dequant s32 -> float (cast only; per-tensor q_scale*k_scale folded into sm_scale
          by the caller), writes s_frag in the C-frag order matching logits_transform,
        - `return`s before the f16 path.
  PV stays fp16 (existing int8->f16 upcast path, untouched).

Layout proven standalone: harness_qk_smem.cu PASS 0/256 vs CPU QK^T using FlashInfer's exact
get_permuted_offset(i,j)=i*stride+(j^(i%8)) k128B swizzle.
"""
import os, flashinfer
FI = os.path.dirname(flashinfer.__file__)
P = os.path.join(FI, "data/include/flashinfer/attention/prefill.cuh")
s = open(P).read()
if not os.path.exists(P + ".i2orig"):
    open(P + ".i2orig", "w").write(s)

# ---- (0) DTypeProb trait: the P (softmax-prob) / PV f16 compute type. For int8 Q this MUST be
# half (NOT int8) — otherwise compute_sfm_v truncates the [0,1] probabilities to int8 -> 0 ->
# softmax denom 0 -> NaN. Decouples the f16 PV math from the int8 storage dtype. ----
TRAIT_OLD = "  using DTypeQ = DTypeQ_;\n  using DTypeKV = DTypeKV_;\n"
TRAIT_NEW = ("  using DTypeQ = DTypeQ_;\n  using DTypeKV = DTypeKV_;\n"
             "  using DTypeProb = std::conditional_t<sizeof(DTypeQ_) == 1, half, DTypeQ_>;\n")
assert TRAIT_OLD in s, "KernelTraits DTypeQ anchor not found"
s = s.replace(TRAIT_OLD, TRAIT_NEW, 1)

# ---- (0b) compute_sfm_v: use DTypeProb for the prob fragment, the float->prob cast, the V
# upcast target, and the PV mma template T (so int8 routes to the .f16 instruction, not .bf16). ----
s = s.replace(
    "  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];",
    "  typename KTraits::DTypeProb s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];", 1)
s = s.replace(
    "        vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],",
    "        vec_cast<typename KTraits::DTypeProb, float>::cast<8>(s_frag_f16[mma_q][mma_kv],", 1)
s = s.replace(
    ("        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(\n"
     "            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_quant);\n"
     "        swap(b_frag[1], b_frag[2]);"),
    ("        vec_cast<typename KTraits::DTypeProb, typename KTraits::DTypeKV>::cast<8>(\n"
     "            (typename KTraits::DTypeProb*)b_frag, (typename KTraits::DTypeKV*)b_frag_quant);\n"
     "        swap(b_frag[1], b_frag[2]);"), 1)
# the two PV mma calls in compute_sfm_v (s_frag_f16 path and s_frag path) -> DTypeProb template T.
s = s.replace(
    ("          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(\n"
     "              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);"),
    ("          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeProb>(\n"
     "              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);"), 1)
s = s.replace(
    ("          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(\n"
     "              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);"),
    ("          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeProb>(\n"
     "              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);"), 1)

# ---- (0c) FIX load_q_global_smem for int8: its head-dim inner loop uses NUM_MMA_D_QK/4 and a
# 2*NUM_MMA_D_QK column-rewind, both HARDCODED for 16-bit geometry (8 threads x 8 elems = 64
# dims/step). For int8 (16 elems/b128) 8 threads cover 128 dims in ONE step, so the f16 count
# does a 2nd OOB step that corrupts Q smem (drops head-dim lanes 4/8/12...). Make it dtype-aware
# like produce_kv (count = NUM_MMA_D_QK/(8/sizeof(DTypeQ)); rewind = sizeof(DTypeQ)*NUM_MMA_D_QK).
# f16 unchanged (8/2=4, 2*8=16); int8 -> count/8, rewind 1*8. ----
s = s.replace(
    "        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {",
    "        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / (8 / sizeof(DTypeQ)); ++mma_do) {", 1)
s = s.replace(
    "        q_smem_offset_w =\n"
    "            q_smem->template advance_offset_by_row<4, UPCAST_STRIDE_Q>(q_smem_offset_w) -\n"
    "            2 * KTraits::NUM_MMA_D_QK;",
    "        q_smem_offset_w =\n"
    "            q_smem->template advance_offset_by_row<4, UPCAST_STRIDE_Q>(q_smem_offset_w) -\n"
    "            sizeof(DTypeQ) * KTraits::NUM_MMA_D_QK;", 1)

# ---- (A) add tid param to compute_qk signature ----
SIG_OLD = ("    uint32_t lane_idx, typename KTraits::DTypeQKAccum "
           "(*s_frag)[KTraits::NUM_MMA_KV][8]) {\n"
           "  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;")
SIG_NEW = ("    uint32_t lane_idx, typename KTraits::DTypeQKAccum "
           "(*s_frag)[KTraits::NUM_MMA_KV][8],\n"
           "    const dim3 tid = threadIdx) {\n"
           "  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;")
assert SIG_OLD in s, "compute_qk signature anchor not found (version drift?)"
s = s.replace(SIG_OLD, SIG_NEW, 1)

# ---- (B) inject int8 branch right after `// compute q*k^T` ... actually right after the
# local-frag decl line inside compute_qk. Anchor on the a_frag/b_frag decl + comment. ----
ANCHOR = ("  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];\n"
          "  // compute q*k^T\n")
assert ANCHOR in s, "compute_qk body anchor not found"

INT8_BRANCH = r'''  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];
  // ===================== I-2: native int8 IMMA QK path (sm_80+) =====================
  // Active when Q is int8. Bypasses ldmatrix; indexes the b128-swizzled int8 smem directly
  // and feeds mma.m16n8k32.row.col.s32.s8.s8.s32. s32 accum -> float. C-frag order matches
  // logits_transform. PV stays fp16.
  //
  // MAGNITUDE NOTE: FlashInfer's mask fill = -math::inf where math::inf is a FINITE 5e4, and
  // masking only works when 5e4 * sm_scale_log2 >> 1. With raw s32 logits (|acc| up to ~1e4)
  // and a tiny per-tensor sm_scale folded by the caller, that condition fails -> causal mask
  // leaks. We therefore pre-scale s32 by INT8_QK_RCP here so s_frag lands in ~f16 magnitude;
  // the caller multiplies sm_scale by INT8_QK_DIV (== 1/INT8_QK_RCP) to keep the softmax exact.
  // (Production per-token dequant applies q_scale*k_scale in-kernel and keeps sm_scale=1/sqrt(d),
  //  so this normalization is a test-harness device, not a numerical change.)
  if constexpr (sizeof(typename KTraits::DTypeQ) == 1) {
    constexpr float INT8_QK_RCP = 1.0f / 256.0f;  // keep paired with i2_test INT8_QK_DIV=256
    using DTypeQ = typename KTraits::DTypeQ;
    using DTypeKV = typename KTraits::DTypeKV;
    constexpr uint32_t HD = KTraits::HEAD_DIM_QK;
    constexpr uint32_t US_Q = KTraits::UPCAST_STRIDE_Q;  // b128 per row (=HD/16)
    constexpr uint32_t US_K = KTraits::UPCAST_STRIDE_K;
    const uint32_t g = lane_idx >> 2, t = lane_idx & 3;
    const uint32_t warp_q_base = get_warp_idx_q<KTraits>(tid.y) * KTraits::NUM_MMA_Q * 16;
    const uint32_t warp_kv_base = get_warp_idx_kv<KTraits>(tid.z) * KTraits::NUM_MMA_KV * 16;
    const int8_t* q_base = reinterpret_cast<const int8_t*>(q_smem->base);
    const int8_t* k_base = reinterpret_cast<const int8_t*>(k_smem->base);
    // logical (row, head-dim col 'dcol' multiple of 4) -> raw u32 of 4 contiguous int8.
    auto ldq = [&](uint32_t row, uint32_t dcol) -> uint32_t {
      uint32_t jb = dcol >> 4, e = dcol & 15;
      uint32_t off = (row * US_Q + (jb ^ (row & 7u))) * 16u + e;
      return *reinterpret_cast<const uint32_t*>(q_base + off);
    };
    auto ldk = [&](uint32_t row, uint32_t dcol) -> uint32_t {
      uint32_t jb = dcol >> 4, e = dcol & 15;
      uint32_t off = (row * US_K + (jb ^ (row & 7u))) * 16u + e;
      return *reinterpret_cast<const uint32_t*>(k_base + off);
    };
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      const uint32_t qr0 = warp_q_base + mma_q * 16 + g;        // Q rows g, g+8
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        int32_t acc[2][4];
#pragma unroll
        for (uint32_t z = 0; z < 8; ++z) acc[z >> 2][z & 3] = 0;
#pragma unroll
        for (uint32_t kd = 0; kd < HD / 32; ++kd) {
          const uint32_t base = kd * 32 + t * 4;
          uint32_t A[4];
          A[0] = ldq(qr0,     base);
          A[1] = ldq(qr0 + 8, base);
          A[2] = ldq(qr0,     base + 16);
          A[3] = ldq(qr0 + 8, base + 16);
#pragma unroll
          for (uint32_t n8 = 0; n8 < 2; ++n8) {
            const uint32_t kr = warp_kv_base + mma_kv * 16 + n8 * 8 + g;  // K row n=g (+8)
            uint32_t B[2];
            B[0] = ldk(kr, base);
            B[1] = ldk(kr, base + 16);
            int32_t* c = acc[n8];
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                  "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
          }
        }
        // s32 -> float into s_frag in C-frag order: reg_id maps as in logits_transform.
        // c0,c1 = (row g)        cols n8*8 + {t*2, t*2+1}
        // c2,c3 = (row g+8)      same cols
        // n8=0 -> reg 0..3 ; n8=1 -> reg 4..7
        typename KTraits::DTypeQKAccum* sf = s_frag[mma_q][mma_kv];
        sf[0] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][0] * INT8_QK_RCP);
        sf[1] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][1] * INT8_QK_RCP);
        sf[2] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][2] * INT8_QK_RCP);
        sf[3] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][3] * INT8_QK_RCP);
        sf[4] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][0] * INT8_QK_RCP);
        sf[5] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][1] * INT8_QK_RCP);
        sf[6] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][2] * INT8_QK_RCP);
        sf[7] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][3] * INT8_QK_RCP);
      }
    }
    return;
  }
  // ===================== end I-2 int8 path =====================
  // compute q*k^T
'''
s = s.replace(ANCHOR, INT8_BRANCH, 1)
open(P, "w").write(s)
print("I2_APPLIED: compute_qk int8 IMMA branch + tid param injected")
