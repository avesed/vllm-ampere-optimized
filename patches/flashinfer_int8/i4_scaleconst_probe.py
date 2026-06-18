#!/usr/bin/env python3
"""I-4a: PRODUCTION int8-QK compute_qk — real per-token in-kernel dequant.

Supersedes i2_compute_qk.py. Apply ON TOP of i1_apply.py + i4_apply.py (scale plumbing).
Run order each container:  i1_apply.py -> i4_apply.py -> i4_compute_qk.py.

Difference vs the I-2 test hack:
  * I-2 wrote   sf[r] = acc * (1/256)   and the caller folded sq*sk into sm_scale and ×256.
    Output magnitude was wrong (|O_i8|/|O_ref|~53) and it only worked for per-TENSOR scales.
  * I-4a writes sf[r] = acc * q_scale[q_tok(r)] * k_scale[kv_tok(r)]  (real per-TOKEN dequant),
    and the caller passes sm_scale = 1/sqrt(d) unchanged (NO folding, NO ×256). s_frag lands in
    natural f16 magnitude => |O_i8|/|O_ref| ~ 1.0 AND FlashInfer's finite mask-fill (-5e4)
    dominates the causal mask (sm_scale_log2 ~ 0.18 for d=128 => exp2(-5e4*0.18)=0).

C-frag -> (q_row, kv_col) mapping (the crux), mirrored from logits_transform (prefill.cuh~L949):
  packed q row = qo_packed_idx_base + mma_q*16 + (lane/4) + 8*((reg%4)/2)   -> group_size.divmod -> q_idx
  kv_idx       = kv_idx_base + mma_kv*16 + 2*(lane%4) + 8*(reg/4) + (reg%2)
With g=lane>>2, t=lane&3 my acc layout is:
  sf[0]=acc[0][0]=(q row g  , kv t*2  )   sf[4]=acc[1][0]=(q row g  , kv 8+t*2  )
  sf[1]=acc[0][1]=(q row g  , kv t*2+1)   sf[5]=acc[1][1]=(q row g  , kv 8+t*2+1)
  sf[2]=acc[0][2]=(q row g+8, kv t*2  )   sf[6]=acc[1][2]=(q row g+8, kv 8+t*2  )
  sf[3]=acc[0][3]=(q row g+8, kv t*2+1)   sf[7]=acc[1][3]=(q row g+8, kv 8+t*2+1)
which is exactly reg_id 0..7 of logits_transform.

Scales are plumbed as fa2 additional tensors maybe_q_scale/maybe_k_scale (float*, in Params),
nullptr-tolerant (fp16/other dtypes pass nullptr => fall back to scale 1.0).
Per-token scalar [num_tokens] (q_idx indexes maybe_q_scale; kv_idx indexes maybe_k_scale,
both already offset to the current request by the call site).
"""
import os, flashinfer
FI = os.path.dirname(flashinfer.__file__)
P = os.path.join(FI, "data/include/flashinfer/attention/prefill.cuh")
s = open(P).read()
if not os.path.exists(P + ".i4orig"):
    open(P + ".i4orig", "w").write(s)

# ---------------------------------------------------------------------------
# (pre) ensure <type_traits>/<utility> for std::void_t / std::declval used by the SFINAE
# helpers below (std::conditional_t already compiles, but make the deps explicit). Inserted
# BEFORE `namespace flashinfer {` so the includes stay at global scope.
# ---------------------------------------------------------------------------
NS_OPEN = "\nnamespace flashinfer {\n"
if "// I-4a includes" not in s:
    assert NS_OPEN in s, "namespace flashinfer open anchor not found"
    s = s.replace(NS_OPEN, "\n#include <type_traits>  // I-4a includes\n#include <utility>\n"
                  + "namespace flashinfer {\n", 1)

# ---------------------------------------------------------------------------
# (0) DTypeProb trait — P (softmax-prob)/PV f16 compute type. int8 Q -> half.
# ---------------------------------------------------------------------------
TRAIT_OLD = "  using DTypeQ = DTypeQ_;\n  using DTypeKV = DTypeKV_;\n"
TRAIT_NEW = ("  using DTypeQ = DTypeQ_;\n  using DTypeKV = DTypeKV_;\n"
             "  using DTypeProb = std::conditional_t<sizeof(DTypeQ_) == 1, half, DTypeQ_>;\n")
assert TRAIT_OLD in s, "KernelTraits DTypeQ anchor not found"
s = s.replace(TRAIT_OLD, TRAIT_NEW, 1)

# ---------------------------------------------------------------------------
# (0b) compute_sfm_v: route prob frag / float->prob cast / V-upcast / PV mma to DTypeProb.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# (0c) load_q_global_smem int8 head-dim geometry fix (dtype-aware count + rewind).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# (A) extend compute_qk signature: add tid + per-token dequant args (all defaulted so the
#     f16 path / other callers are unaffected; the int8 path reads them).
# ---------------------------------------------------------------------------
SIG_OLD = ("    uint32_t lane_idx, typename KTraits::DTypeQKAccum "
           "(*s_frag)[KTraits::NUM_MMA_KV][8]) {\n"
           "  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;")
SIG_NEW = ("    uint32_t lane_idx, typename KTraits::DTypeQKAccum "
           "(*s_frag)[KTraits::NUM_MMA_KV][8],\n"
           "    const dim3 tid = threadIdx,\n"
           "    const float* q_dequant_scale = nullptr,\n"
           "    const float* k_dequant_scale = nullptr,\n"
           "    const uint32_t qo_packed_idx_base = 0, const uint32_t kv_idx_base = 0,\n"
           "    const uint_fastdiv group_size = uint_fastdiv(),\n"
           "    const uint32_t qo_len = 0, const uint32_t kv_len = 0) {\n"
           "  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;")
assert SIG_OLD in s, "compute_qk signature anchor not found (version drift?)"
s = s.replace(SIG_OLD, SIG_NEW, 1)

# ---------------------------------------------------------------------------
# (B) inject the int8 IMMA branch with REAL per-token dequant.
# ---------------------------------------------------------------------------
ANCHOR = ("  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];\n"
          "  // compute q*k^T\n")
assert ANCHOR in s, "compute_qk body anchor not found"

INT8_BRANCH = r'''  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];
  // ============== I-4a: native int8 IMMA QK + real per-token dequant (sm_80+) ==============
  // Active when Q is int8. Bypasses ldmatrix; indexes the b128-swizzled int8 smem directly,
  // feeds mma.m16n8k32.row.col.s32.s8.s8.s32, then dequantizes s32 -> float with
  //   sf[r] = acc * q_dequant_scale[q_idx(r)] * k_dequant_scale[kv_idx(r)]
  // (per-token symmetric int8 scales). The caller keeps sm_scale = 1/sqrt(d) (applied
  // downstream in update_mdo_states), so s_frag has natural f16 magnitude and the finite
  // mask-fill (-5e4) dominates the causal mask. nullptr scales => factor 1.0 (no-op).
  // C-frag (q_row,kv_col) mirrors logits_transform reg_id 0..7. PV stays fp16.
  if constexpr (sizeof(typename KTraits::DTypeQ) == 1) {
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
    // Per-fragment absolute q-token indices (two q rows: g, g+8) via group_size.divmod,
    // exactly as logits_transform de-groups the packed index for GQA.
    // IMPORTANT: a kv/q tile of width NUM_MMA*16 can extend PAST kv_len/qo_len (the tail
    // is masked out by logits_mask AFTER compute_qk). We must NOT read the scale tensor OOB
    // for those padding rows/cols (garbage scale -> huge logit -> exp2 overflow before the
    // mask fires). Bound the index; the value is irrelevant (it gets masked), so use 1.0.
    auto qscale = [&](uint32_t packed_q) -> float {
      if (q_dequant_scale == nullptr) return 1.0f;
      uint32_t q_idx, r_unused;
      group_size.divmod(packed_q, q_idx, r_unused);
      return (q_idx < qo_len) ? __ldg(q_dequant_scale + q_idx) : 1.0f;
    };
    auto kscale = [&](uint32_t kv_idx) -> float {
      if (k_dequant_scale == nullptr) return 1.0f;
      return (kv_idx < kv_len) ? __ldg(k_dequant_scale + kv_idx) : 1.0f;
    };
    // Precompute scales ONCE (perf): q-scale depends only on mma_q, k-scale only on mma_kv+t.
    // Avoids re-reading gmem / re-doing group_size.divmod inside the hot mma loops.
    float qs_lo_a[KTraits::NUM_MMA_Q], qs_hi_a[KTraits::NUM_MMA_Q]; (void)qscale;
#pragma unroll
    for (uint32_t mq = 0; mq < KTraits::NUM_MMA_Q; ++mq) {
      qs_lo_a[mq] = 1.0f;
      qs_hi_a[mq] = 1.0f;
    }
    float ks0a_a[KTraits::NUM_MMA_KV], ks0b_a[KTraits::NUM_MMA_KV];
    float ks1a_a[KTraits::NUM_MMA_KV], ks1b_a[KTraits::NUM_MMA_KV];
#pragma unroll
    for (uint32_t mk = 0; mk < KTraits::NUM_MMA_KV; ++mk) {
      const uint32_t kv0 = kv_idx_base + mk * 16 + 2 * t;
      const uint32_t kv1 = kv_idx_base + mk * 16 + 8 + 2 * t;
      ks0a_a[mk] = 1.0f; ks0b_a[mk] = 1.0f;
      ks1a_a[mk] = 1.0f; ks1b_a[mk] = 1.0f; (void)kscale;
    }
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      const uint32_t qr0 = warp_q_base + mma_q * 16 + g;        // Q smem rows g, g+8
      const float qs_lo = qs_lo_a[mma_q];
      const float qs_hi = qs_hi_a[mma_q];
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
            const uint32_t kr = warp_kv_base + mma_kv * 16 + n8 * 8 + g;  // K smem row
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
        const float ks0a = ks0a_a[mma_kv], ks0b = ks0b_a[mma_kv];
        const float ks1a = ks1a_a[mma_kv], ks1b = ks1b_a[mma_kv];
        // Direct per-token dequant. Out-of-range tile padding (q_idx>=qo_len / kv_idx>=kv_len)
        // reads scale 1.0 (qscale/kscale bound the deref) -> a raw |acc| logit there, which
        // logits_mask THEN overwrites with MaskFillValue (kv_idx>=chunk_end and the causal/q
        // bound are all masked downstream before softmax). Cheap: no per-element branches.
        typename KTraits::DTypeQKAccum* sf = s_frag[mma_q][mma_kv];
        sf[0] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][0] * qs_lo * ks0a);
        sf[1] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][1] * qs_lo * ks0b);
        sf[2] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][2] * qs_hi * ks0a);
        sf[3] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[0][3] * qs_hi * ks0b);
        sf[4] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][0] * qs_lo * ks1a);
        sf[5] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][1] * qs_lo * ks1b);
        sf[6] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][2] * qs_hi * ks1a);
        sf[7] = static_cast<typename KTraits::DTypeQKAccum>((float)acc[1][3] * qs_hi * ks1b);
      }
    }
    return;
  }
  // ===================== end I-4a int8 path =====================
  // compute q*k^T
'''
s = s.replace(ANCHOR, INT8_BRANCH, 1)

# ---------------------------------------------------------------------------
# (C) call sites: pass kv_idx_base + scales + index/grouping info into compute_qk.
#     Each of the 3 sites already computes kv_idx_base on the NEXT line; hoist it ABOVE the
#     compute_qk call (compute it first), then forward the dequant args. The maybe_q_scale /
#     maybe_k_scale fields exist in Params only for int8 modules (i4_apply) and are nullptr-
#     tolerant; for f16 modules the fields are absent so we guard with has_* via the SFINAE
#     helpers FlashInfer already uses... but simpler: the fields are ALWAYS declared by the
#     codegen list, so we reference params.maybe_q_scale directly (present iff i4_apply added
#     them; the f16 module just never gets the int8 branch). To keep f16 modules compiling
#     (which DON'T have the fields), we read the scale ptrs through a constexpr-guarded helper.
# ---------------------------------------------------------------------------
# Single prefill (batch_idx 0): qo_packed_idx_base/qo_len/kv_len/group_size/kv_head_idx in scope.
CALL_SINGLE_OLD = (
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag);\n"
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;")
CALL_SINGLE_NEW = (
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;\n"
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag, tid,\n"
    "                          get_q_dequant_scale<KTraits>(params, 0),\n"
    "                          get_k_dequant_scale<KTraits>(params, 0),\n"
    "                          qo_packed_idx_base, kv_idx_base, group_size, qo_len, kv_len);")
assert CALL_SINGLE_OLD in s, "single compute_qk call site anchor not found"
s = s.replace(CALL_SINGLE_OLD, CALL_SINGLE_NEW, 1)

# Batch ragged (batch_idx request_idx).
CALL_RAGGED_OLD = (
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag);\n"
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;\n"
    "      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,")
CALL_RAGGED_NEW = (
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;\n"
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag, tid,\n"
    "                          get_q_dequant_scale<KTraits>(params, request_idx),\n"
    "                          get_k_dequant_scale<KTraits>(params, request_idx),\n"
    "                          qo_packed_idx_base, kv_idx_base, group_size, qo_len, kv_len);\n"
    "      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,")
assert CALL_RAGGED_OLD in s, "ragged compute_qk call site anchor not found"
s = s.replace(CALL_RAGGED_OLD, CALL_RAGGED_NEW, 1)

# Batch paged (batch_idx request_idx).
CALL_PAGED_OLD = (
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag);\n"
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;\n"
    "      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,")
CALL_PAGED_NEW = (
    "      uint32_t kv_idx_base =\n"
    "          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;\n"
    "      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,\n"
    "                          smem_storage.k_sf_smem + get_warp_idx_kv<KTraits>(tid.z) *\n"
    "                                                       KTraits::NUM_MMA_KV * 16 *\n"
    "                                                       KTraits::NUM_MMA_D_QK,\n"
    "                          lane_idx, s_frag, tid,\n"
    "                          get_q_dequant_scale<KTraits>(params, request_idx),\n"
    "                          get_k_dequant_scale<KTraits>(params, request_idx),\n"
    "                          qo_packed_idx_base, kv_idx_base, group_size, qo_len, kv_len);\n"
    "      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,")
# the ragged replace already consumed one occurrence; paged is identical text, replace remaining.
assert s.count(CALL_PAGED_OLD) >= 1, "paged compute_qk call site anchor not found"
s = s.replace(CALL_PAGED_OLD, CALL_PAGED_NEW, 1)

# ---------------------------------------------------------------------------
# (D) constexpr-guarded scale-pointer accessors. f16 Params do NOT declare maybe_q_scale /
#     maybe_k_scale -> use SFINAE has_member detection so the same call sites compile for both.
#     For int8 Params these fields are float* (added by i4_apply); for batch we offset by the
#     request's q_indptr / a per-request kv base. For single (one request) offset is 0.
#     The q-scale tensor is laid out [total_qo_tokens]; q_indptr gives the request base.
#     The k-scale tensor is laid out [total_kv_tokens]; we offset by params.kv_scale_indptr
#     if present, else 0 (single / ragged single-request tests pass request-local tensors).
# ---------------------------------------------------------------------------
HELPER = r'''
// ---- I-4a: SFINAE-guarded per-token dequant scale accessors (int8 modules only). NOTE: this
// block is injected INSIDE the existing `namespace flashinfer { ... }` of prefill.cuh, so it
// must NOT re-open the namespace (doing so makes flashinfer::flashinfer and breaks ::Error etc).
template <typename T, typename = void>
struct i4_has_maybe_q_scale : std::false_type {};
template <typename T>
struct i4_has_maybe_q_scale<T, std::void_t<decltype(std::declval<T>().maybe_q_scale)>>
    : std::true_type {};
template <typename T, typename = void>
struct i4_has_q_indptr : std::false_type {};
template <typename T>
struct i4_has_q_indptr<T, std::void_t<decltype(std::declval<T>().q_indptr)>>
    : std::true_type {};
template <typename T, typename = void>
struct i4_has_kv_scale_indptr : std::false_type {};
template <typename T>
struct i4_has_kv_scale_indptr<T, std::void_t<decltype(std::declval<T>().maybe_kv_scale_indptr)>>
    : std::true_type {};

template <typename KTraits, typename Params>
__device__ __forceinline__ const float* get_q_dequant_scale(const Params& params,
                                                            uint32_t request_idx) {
  if constexpr (i4_has_maybe_q_scale<Params>::value) {
    if (params.maybe_q_scale == nullptr) return nullptr;
    uint32_t off = 0;
    if constexpr (i4_has_q_indptr<Params>::value) {
      off = params.q_indptr[request_idx];
    }
    return params.maybe_q_scale + off;
  } else {
    return nullptr;
  }
}
template <typename KTraits, typename Params>
__device__ __forceinline__ const float* get_k_dequant_scale(const Params& params,
                                                            uint32_t request_idx) {
  if constexpr (i4_has_maybe_q_scale<Params>::value) {
    if (params.maybe_k_scale == nullptr) return nullptr;
    uint32_t off = 0;
    if constexpr (i4_has_kv_scale_indptr<Params>::value) {
      off = params.maybe_kv_scale_indptr[request_idx];
    }
    return params.maybe_k_scale + off;
  } else {
    return nullptr;
  }
}
'''
# insert the helper just before compute_qk's template line so it's in scope at the call sites
# (call sites are AFTER compute_qk in the file, and compute_qk references nothing from helper).
INSERT_ANCHOR = "template <typename KTraits>\n__device__ __forceinline__ void compute_qk("
assert INSERT_ANCHOR in s, "compute_qk insert anchor not found"
s = s.replace(INSERT_ANCHOR, HELPER + "\n" + INSERT_ANCHOR, 1)

open(P, "w").write(s)
print("I4_SCALECONST_PROBE_APPLIED: per-token dequant int8 compute_qk + 3 call sites + scale accessors")
