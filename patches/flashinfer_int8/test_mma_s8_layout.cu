// I-2 foundation (PASS 0/128 vs CPU): validates mma.m16n8k32.row.col.s32.s8.s8.s32 wrapper +
// the exact PTX fragment/thread layout used to wire int8-QK into FlashInfer compute_qk.
// A[16,32] s8 row-major; B[8,32] s8 row-major (=operand B[k][n], i.e. B'[n][k]); C[16,8] s32.
// lane: g=lane>>2 (0..7), t=lane&3. A frag a0..a3 = A[g|g+8][t*4 + {0..3}|{16..19}].
// B frag b0,b1 = B[n=g][k=t*4 + {0..3}|{16..19}]. C c0..c3 = C[g|g+8][t*2 + {0,1}].
// (full kernel in ~/test_mma_num.cu on sandbox; nvcc -arch=sm_86, PASS on RTX 3090)
