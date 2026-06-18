// Standalone harness: replicate FlashInfer FA2 int8 Q/K swizzled smem layout and validate a
// candidate int8 m16n8k32 fragment-load (for compute_qk I-2) against a CPU QK^T reference.
//
// Config mirrors the test: HEAD_DIM=128 (int8), one CTA tile of 16 Q tokens x 16 KV tokens,
// single warp (NUM_WARPS_Q=NUM_WARPS_KV=1), NUM_MMA_Q=1, NUM_MMA_KV=1, NUM_MMA_D_QK=8.
// upcast_size<int8>()=16 -> UPCAST_STRIDE = HEAD_DIM/16 = 8 (b128 per row).
//
// Smem layout (from load_q_global_smem / produce_kv + get_permuted_offset k128B):
//   smem is b128_t[ROWS][UPCAST_STRIDE]; row r = token r; b128 column j holds head-dim int8
//   elements [j*16 .. j*16+15]; physical column index = (j ^ (r % 8)).
//
// We build smem in HOST memory exactly that way, copy to device smem-shaped global buffer, and a
// kernel reads it per-thread to form the m16n8k32 fragments, runs the validated s8s8s32 mma, and
// dequant-free compares the s32 result vs CPU int32 QK^T.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define HEAD_DIM 128
#define UPCAST_STRIDE 8     // HEAD_DIM / 16
#define ROWS_Q 16
#define ROWS_K 16
#define NUM_MMA_D_QK 8
#define NUM_MMA_D_K32 (NUM_MMA_D_QK / 2)   // = 4 ; m16n8k32 tiles over head_dim

// permuted offset (in b128 units) for k128B swizzle, stride=UPCAST_STRIDE.
__host__ __device__ inline uint32_t perm(uint32_t i, uint32_t j) {
  return i * UPCAST_STRIDE + (j ^ (i % 8));
}

// Each b128 = 16 int8. Build smem[row][headdim] -> physical b128 buffer.
// buf is int8_t[ROWS * UPCAST_STRIDE * 16].
void build_smem(const int8_t* mat /*[ROWS][HEAD_DIM] row-major*/, int rows, int8_t* buf) {
  for (int r = 0; r < rows; r++)
    for (int j = 0; j < UPCAST_STRIDE; j++) {
      uint32_t off = perm(r, j);            // physical b128 index
      for (int e = 0; e < 16; e++)
        buf[off * 16 + e] = mat[r * HEAD_DIM + j * 16 + e];
    }
}

// ---- the candidate fragment load lives HERE. We index the swizzled smem directly. ----
// q smem read base offset (compute_qk init): get_permuted_offset(lane%16, lane/16).
// k smem read base offset:                   get_permuted_offset(8*(lane/16) + lane%8, (lane%16)/8).
// But we BYPASS that and index by logical (token,headdim) via perm(), reproducing the validated
// m16n8k32 layout:  a0..a3 = Q[g|g+8][col], b0,b1 = K[n=g][col], col = base + t*4 + {0..3 | 16..19}
// where for k32 tile 'kd' (0..3), the head-dim base = kd*32.

__global__ void qk_kernel(const int8_t* q_smem, const int8_t* k_smem, int32_t* C /*[16][16]*/) {
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  int32_t acc[2][4];  // two n8 tiles (n=0..7, n=8..15) x 4 s32 each
  #pragma unroll
  for (int n8 = 0; n8 < 2; n8++)
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[n8][i] = 0;

  auto ld4 = [](const int8_t* smem, int row, int dcol) -> uint32_t {
    // load 4 contiguous int8 at logical (row, head-dim dcol..dcol+3) from swizzled smem.
    // dcol is multiple of 4 within a 16-elem b128 -> all 4 in one b128.
    uint32_t jb = dcol / 16;           // b128 column
    uint32_t e  = dcol % 16;           // element offset in b128
    uint32_t off = perm(row, jb) * 16 + e;
    return *(const uint32_t*)(smem + off);
  };

  #pragma unroll
  for (int kd = 0; kd < NUM_MMA_D_K32; kd++) {
    int base = kd * 32;
    // A frag (Q): rows g, g+8 ; cols base+t*4+{0..3}, base+t*4+16+{0..3}
    uint32_t A[4];
    A[0] = ld4(q_smem, g,     base + t * 4);
    A[1] = ld4(q_smem, g + 8, base + t * 4);
    A[2] = ld4(q_smem, g,     base + t * 4 + 16);
    A[3] = ld4(q_smem, g + 8, base + t * 4 + 16);
    #pragma unroll
    for (int n8 = 0; n8 < 2; n8++) {
      int nrow = n8 * 8 + g;            // K token row n=g (+8 for second n8 tile)
      uint32_t B[2];
      B[0] = ld4(k_smem, nrow, base + t * 4);
      B[1] = ld4(k_smem, nrow, base + t * 4 + 16);
      int32_t* c = acc[n8];
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(c[0]),"=r"(c[1]),"=r"(c[2]),"=r"(c[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),
          "r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
    }
  }
  // C frag: c0..c3 = C[g|g+8][n8col + t*2 + {0,1}], n8col = n8*8.
  #pragma unroll
  for (int n8 = 0; n8 < 2; n8++) {
    int32_t* c = acc[n8];
    C[(g)     * 16 + n8 * 8 + t * 2 + 0] = c[0];
    C[(g)     * 16 + n8 * 8 + t * 2 + 1] = c[1];
    C[(g + 8) * 16 + n8 * 8 + t * 2 + 0] = c[2];
    C[(g + 8) * 16 + n8 * 8 + t * 2 + 1] = c[3];
  }
}

int main() {
  int8_t hQ[ROWS_Q * HEAD_DIM], hK[ROWS_K * HEAD_DIM];
  for (int i = 0; i < ROWS_Q * HEAD_DIM; i++) hQ[i] = (int8_t)((i * 7 + 3) % 13 - 6);
  for (int i = 0; i < ROWS_K * HEAD_DIM; i++) hK[i] = (int8_t)((i * 5 + 1) % 11 - 5);
  // CPU ref: C[m][n] = sum_k Q[m][k]*K[n][k]
  int32_t ref[ROWS_Q * ROWS_K];
  for (int m = 0; m < ROWS_Q; m++)
    for (int n = 0; n < ROWS_K; n++) {
      int s = 0;
      for (int k = 0; k < HEAD_DIM; k++) s += (int)hQ[m * HEAD_DIM + k] * (int)hK[n * HEAD_DIM + k];
      ref[m * ROWS_K + n] = s;
    }
  int8_t qbuf[ROWS_Q * UPCAST_STRIDE * 16], kbuf[ROWS_K * UPCAST_STRIDE * 16];
  build_smem(hQ, ROWS_Q, qbuf);
  build_smem(hK, ROWS_K, kbuf);

  int8_t *dQ, *dK; int32_t *dC;
  cudaMalloc(&dQ, sizeof qbuf); cudaMalloc(&dK, sizeof kbuf); cudaMalloc(&dC, sizeof(int32_t) * 256);
  cudaMemcpy(dQ, qbuf, sizeof qbuf, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, kbuf, sizeof kbuf, cudaMemcpyHostToDevice);
  qk_kernel<<<1, 32>>>(dQ, dK, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) { printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; }
  int32_t hC[256];
  cudaMemcpy(hC, dC, sizeof(int32_t) * 256, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int m = 0; m < ROWS_Q; m++)
    for (int n = 0; n < ROWS_K; n++)
      if (hC[m * 16 + n] != ref[m * 16 + n]) {
        if (bad < 8) printf("MISMATCH[%d,%d] gpu=%d ref=%d\n", m, n, hC[m * 16 + n], ref[m * 16 + n]);
        bad++;
      }
  printf("%s (%d/256 mismatch)\n", bad ? "FAIL" : "PASS", bad);
  return 0;
}
