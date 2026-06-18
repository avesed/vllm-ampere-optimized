// Host simulator of FlashInfer FA2 int8 K/Q smem swizzle WRITE (produce_kv / load_q_global_smem)
// vs my compute_qk int8 READ, to find the head-dim addressing bug deterministically (no GPU/JIT).
// Models head_dim=128 int8, k128B swizzle, single CTA. Verifies: physical byte written for logical
// (token, dim) == physical byte my read fetches for the same (token, dim).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

// config (mirror a representative case: cta_tile_q=128 -> NUM_WARPS_Q=4,NUM_WARPS_KV=1,NUM_MMA_Q=2;
// for KV: CTA_TILE_KV=128 -> NUM_MMA_KV=8). We test K (produce_kv) and Q (load_q) writes.
const int HEAD_DIM = 128;
const int UPCAST = 16;                 // int8 per b128
const int US = HEAD_DIM / UPCAST;      // = 8  (b128 per row)
const int WARP=32;

// get_permuted_offset k128B (b128 units): i*US + (j ^ (i%8))
inline int perm(int i, int j){ return i*US + (j ^ (i%8)); }

// advance_offset_by_column<step> k128B
inline int adv_col(int off, int step, int step_idx){
  if(step==2) return (off ^ (0x2 + (0x4*(step_idx%2==1)))) + (step_idx%4==3)*8;
  if(step==4) return (off ^ 0x4) + (step_idx%2==1)*8;
  return off + step; // step%8==0
}
inline int adv_row(int off, int step, int row_stride){
  if(step==4) return (off ^ 0x4) + step*row_stride;
  return off + step*row_stride; // step%8==0
}

// ---- Simulate produce_kv<false> (K), k128B branch. NUM_MMA_KV, NUM_WARPS_Q,KV. ----
// Writes, per (warp,lane): kv tokens and head-dim chunks. Records phys b128 offset for (token,chunk).
struct Map { int phys[256][US]; }; // phys[token][b128_chunk] = physical b128 offset (or -1)

void sim_produce_kv(int NUM_MMA_KV,int NUM_WARPS_Q,int NUM_WARPS_KV,Map& M){
  int NUM_WARPS=NUM_WARPS_Q*NUM_WARPS_KV;
  int NUM_MMA_D = US;                  // produce K: NUM_MMA_D_QK = HEAD_DIM/16 = US... actually NUM_MMA_D_QK=HEAD_DIM/16=8
  int CTA_TILE_KV = NUM_MMA_KV*NUM_WARPS_KV*16;
  memset(M.phys,-1,sizeof(M.phys));
  // KV_THR_LAYOUT_ROW=4, COL=8 (k128B). smem_offset_w init = perm(warp*4+lane/8, lane%8)
  for(int warp=0; warp<NUM_WARPS; warp++)
  for(int lane=0; lane<WARP; lane++){
    int off = perm(warp*4 + lane/8, lane%8);
    int kv_idx = warp*4 + lane/8;     // kv_idx_base=0
    for(int i=0;i< NUM_MMA_KV*4/NUM_WARPS_Q; i++){
      for(int j=0;j< NUM_MMA_D/(8/1); j++){       // NUM_MMA_D/(8/sizeof int8=8) = NUM_MMA_D/8
        // load_128b_async to off : writes 16 int8 = one head-dim b128 chunk.
        // which head-dim chunk? gptr advances by 8*upcast each j -> chunk index = j*8 + ... hmm.
        // Actually each thread's gptr starts at (lane%8)*upcast = head-dim element (lane%8)*16.
        // chunk = lane%8 + j*8? gptr += 8*upcast_size each j. So chunk col index = (lane%8) + j*8.
        int chunk = (lane%8) + j*8;
        if(kv_idx < 256 && chunk < US) M.phys[kv_idx][chunk] = off;
        off = adv_col(off, 8, j);
      }
      kv_idx += NUM_WARPS*4;
      off = adv_row(off, NUM_WARPS*4, US) - 1*NUM_MMA_D;   // sizeof(int8)=1
    }
  }
}

// ---- my READ: for kv token 'row', head-dim chunk 'jb', phys = perm(row, jb) ----
int my_read_phys(int row,int jb){ return perm(row, jb); }

// ---- Simulate load_q_global_smem (Q), k128B. NUM_MMA_Q, NUM_WARPS_Q. ----
void sim_load_q(int NUM_MMA_Q,int NUM_WARPS_Q,Map& M){
  int NUM_MMA_D_QK = US; // HEAD_DIM/16 = 8
  memset(M.phys,-1,sizeof(M.phys));
  for(int warp_x=0; warp_x<NUM_WARPS_Q; warp_x++)
  for(int lane=0; lane<WARP; lane++){
    int off = perm(warp_x*NUM_MMA_Q*16 + lane/8, lane%8);
    for(int mma_q=0; mma_q<NUM_MMA_Q; mma_q++){
      for(int j=0;j<2*2;j++){
        int token = warp_x*NUM_MMA_Q*16 + lane/8 + mma_q*16 + j*4;  // smem row = packed token (gs=1)
        // q_ptr starts at (lane%8)*upcast = head-dim element (lane%8)*16  => chunk lane%8
        for(int mma_do=0; mma_do<NUM_MMA_D_QK/4; mma_do++){
          int chunk = (lane%8) + mma_do*8;
          if(token<256 && chunk<US) M.phys[token][chunk] = off;
          off = adv_col(off, 8, mma_do);
        }
        off = adv_row(off, 4, US) - 2*NUM_MMA_D_QK;
      }
    }
  }
}

int main(){
  Map M;
  sim_produce_kv(8, 4, 1, M);
  { int bad=0,checked=0;
    for(int tok=0;tok<128;tok++)for(int jb=0;jb<US;jb++){int w=M.phys[tok][jb];if(w<0)continue;int r=my_read_phys(tok,jb);checked++;if(w!=r){if(bad<8)printf("K MISMATCH tok=%d jb=%d w=%d r=%d\n",tok,jb,w,r);bad++;}}
    printf("Case A (K, MMA_KV=8,WQ=4,WKV=1): checked=%d bad=%d\n",checked,bad);
  }
  sim_load_q(2, 4, M);   // cta_tile_q=128: NUM_MMA_Q=2, NUM_WARPS_Q=4
  { int bad=0,checked=0;
    for(int tok=0;tok<128;tok++)for(int jb=0;jb<US;jb++){int w=M.phys[tok][jb];if(w<0)continue;int r=my_read_phys(tok,jb);checked++;if(w!=r){if(bad<12)printf("Q MISMATCH tok=%d jb=%d w=%d r=%d (dim chunk diff)\n",tok,jb,w,r);bad++;}}
    printf("Case Q (load_q, MMA_Q=2,WQ=4): checked=%d bad=%d\n",checked,bad);
  }
  return 0;
}
