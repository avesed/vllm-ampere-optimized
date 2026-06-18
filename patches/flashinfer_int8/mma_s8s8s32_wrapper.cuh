// I-0 (validated): int8 IMMA wrapper for Ampere sm_80+, to be added to FlashInfer's
// data/include/flashinfer/mma.cuh (clone of mma_sync_m16n16k32_row_col_f8f8f32 @ ~line 217;
// PTX f32.e4m3.e4m3.f32 -> s32.s8.s8.s32, output regs "=f"->"=r", C-accum float->int32).
//
// STATUS: compiles clean for sm_86 (nvcc -arch=sm_86 -c, RC=0, 20 regs, 0 spill); -ptx confirms
// the two m16n8k32 s8s8s32 instructions emit correctly (C-input = RZ for kInit). Numerical
// correctness to be validated when wired into compute_qk (I-2), output checked vs fp16.
//
// C: 8x int32 per thread (m16n16 = two m16n8 tiles). A: 4x uint32 (16 int8). B: 4x uint32 (2 per n8 tile).

template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
  static_assert(sizeof(T) == 1, "int8 mma operand must be 8-bit");
#if defined(FLASHINFER_MMA_S8S8S32_M16N8K32_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(0), "r"(0), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
          "r"(0), "r"(0), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
          "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
  }
#else
  static_assert(sizeof(T) == 0, "s8s8s32 m16n8k32 requires FLASHINFER_MMA_S8S8S32_M16N8K32_ENABLED (sm_80+)");
#endif
}
