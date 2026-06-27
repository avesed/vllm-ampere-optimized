// Fused silu_and_mul + per-token int8 quant for the W4A8 FFN (Ampere sm_80/sm_86).
//
// Replaces, in ONE launch:
//   gate_up(marlin) -> SiluAndMul (bf16 kernel, HBM round-trip) -> per_token_quant_int8 (Triton)
// The bf16 [M,N] intermediate never leaves SMEM/registers. PREFILL lever (M large); decode ~0.
//
// Bit-match target = the TRITON _per_token_quant_int8 (int8_utils.py:105-128), NOT the C++ kernel:
//   absmax = max(|y|) over the row, fp32; absmax = max(absmax, 1e-10)
//   scale  = absmax / 127.0          (TRUE divide, stored to scales[m])
//   inv    = 127.0 / absmax          (reciprocal form, NOT 1/scale)
//   q      = roundf(y * inv)         (round-half-AWAY-from-zero == CUDA __nv_round
//                                     == Triton libdevice.round; NOT rintf/half-to-even)
//   out    = (int8) clamp(q, -127, 127)
// and the bf16 round-trip of the activation (silu in fp32 -> bf16, * up in bf16) is reproduced
// by staging y as bf16, then casting bf16->fp32 for absmax/quant.

#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBlock = 256;          // 8 warps
constexpr int kWarp = 32;
constexpr int kMaxWarps = kBlock / kWarp;
// block_reduce_max assumes the block is a whole number of warps and fits one warp of warp-maxima.
static_assert(kBlock % kWarp == 0 && kBlock / kWarp <= kWarp,
              "kBlock must be a whole number of warps and <= 1024 threads");
constexpr float kEps = 1e-10f;       // absmax floor (Triton); NOT 1e-12, NOT 0
// SMEM reserved by the staged kernel beyond the dynamic y[] buffer: the static s_warp_max[]
// plus a small driver margin. The per-block HW limit covers dynamic+static combined, and the
// driver may reserve ~1KB even within the opt-in number, so the dynamic request and the stage
// threshold must both leave this much headroom under the opt-in cap.
constexpr int kStaticSmem = (int)(sizeof(float) * kMaxWarps);
constexpr int kSmemReserve = kStaticSmem + 1024;

// fp32 silu, plain divide + expf (matches silu_kernel / SiluAndMul; NOT __fdividef/__expf).
__device__ __forceinline__ float silu_f(float g) {
  return g / (1.0f + expf(-g));
}

// scalar_t conversions to/from fp32 for the two supported input dtypes.
__device__ __forceinline__ float to_f(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(const __half v) { return __half2float(v); }
__device__ __forceinline__ __nv_bfloat16 from_f(float v, __nv_bfloat16*) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half from_f(float v, __half*) { return __float2half_rn(v); }

// Compute the per-row activation y = silu(gate) * up, reproducing the reference bf16/fp16
// round-trip: silu in fp32 -> round to scalar_t, multiply by up in scalar_t, then read back as fp32.
template <typename scalar_t>
__device__ __forceinline__ float act_elem(scalar_t g_raw, scalar_t u_raw) {
  float gf = to_f(g_raw);
  scalar_t* tag = nullptr;
  scalar_t s = from_f(silu_f(gf), tag);   // silu_fp32 -> scalar_t (bf16/fp16 rounding)
  scalar_t y = s * u_raw;                  // multiply in scalar_t (operator* -> __hmul)
  return to_f(y);                          // read back exactly what the reference staged in HBM
}

// Two-level block max reduction of a per-thread value; returns the block max on every thread.
__device__ __forceinline__ float block_reduce_max(float v, float* s_warp_max) {
  // intra-warp
  #pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) s_warp_max[warp] = v;
  __syncthreads();
  // cross-warp: warp 0 reduces the per-warp maxima, broadcasts via s_warp_max[0]
  if (warp == 0) {
    float w = (lane < (blockDim.x >> 5)) ? s_warp_max[lane] : 0.0f;
    #pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
      w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, off));
    if (lane == 0) s_warp_max[0] = w;
  }
  __syncthreads();
  return s_warp_max[0];
}

__device__ __forceinline__ int8_t quantize(float y, float inv) {
  // roundf == round-half-AWAY-from-zero == CUDA __nv_round == Triton libdevice.round.
  float q = roundf(y * inv);
  q = fminf(fmaxf(q, -127.0f), 127.0f);           // defensive clamp to [-127,127]
  if (!isfinite(q)) q = 0.0f;                      // make NaN/Inf -> int8 store well-defined
  return static_cast<int8_t>(q);
}

// SMEM-staged kernel: stage y (scalar_t) for the whole row, reduce absmax, then quantize from SMEM.
// One block == one full token-row so the per-row absmax is exact.
template <typename scalar_t>
__global__ void staged_kernel(const scalar_t* __restrict__ input,
                              int8_t* __restrict__ out,
                              float* __restrict__ scales,
                              int N) {
  extern __shared__ __align__(16) unsigned char s_raw[];
  scalar_t* s_y = reinterpret_cast<scalar_t*>(s_raw);
  __shared__ float s_warp_max[kMaxWarps];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* g = input + (size_t)row * 2 * N;   // gate = [0:N]
  const scalar_t* u = g + N;                          // up   = [N:2N]

  // Phase 1: compute y, stage as scalar_t, track per-thread absmax.
  float local_max = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    scalar_t* tag = nullptr;
    float yf = act_elem(g[i], u[i]);
    s_y[i] = from_f(yf, tag);
    local_max = fmaxf(local_max, fabsf(yf));
  }

  // Phase 1.5: block absmax -> scale.
  float absmax = fmaxf(block_reduce_max(local_max, s_warp_max), kEps);
  float scale = absmax / 127.0f;      // TRUE divide; bit-matches Triton scale_x = absmax/127
  float inv = 127.0f / absmax;        // reciprocal form for the quant multiply (matches Triton)
  if (tid == 0) scales[row] = scale;

  // Phase 2: re-read y from SMEM, quantize, write int8.
  int8_t* out_row = out + (size_t)row * N;
  for (int i = tid; i < N; i += blockDim.x) {
    out_row[i] = quantize(to_f(s_y[i]), inv);
  }
}

// Recompute fallback (N too large to stage): no SMEM; recompute y in phase 2 from HBM.
// Still avoids the bf16 [M,N] HBM *output* round-trip (only re-reads the inputs).
template <typename scalar_t>
__global__ void recompute_kernel(const scalar_t* __restrict__ input,
                                 int8_t* __restrict__ out,
                                 float* __restrict__ scales,
                                 int N) {
  __shared__ float s_warp_max[kMaxWarps];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* g = input + (size_t)row * 2 * N;
  const scalar_t* u = g + N;

  float local_max = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(act_elem(g[i], u[i])));
  }

  float absmax = fmaxf(block_reduce_max(local_max, s_warp_max), kEps);
  float scale = absmax / 127.0f;      // TRUE divide; bit-matches Triton scale_x = absmax/127
  float inv = 127.0f / absmax;
  if (tid == 0) scales[row] = scale;

  int8_t* out_row = out + (size_t)row * N;
  for (int i = tid; i < N; i += blockDim.x) {
    out_row[i] = quantize(act_elem(g[i], u[i]), inv);
  }
}

template <typename scalar_t>
void launch(const scalar_t* in, int8_t* out, float* scales,
            int M, int N, size_t smem, bool stage, cudaStream_t stream) {
  if (stage) {
    if (smem > 48u * 1024u) {
      // opt-in dynamic SMEM above the 48KB static default (sm_86 ~100KB, sm_80 ~163KB).
      // The caller already guaranteed smem <= cap - kSmemReserve, so this never crowds out the
      // static s_warp_max[]; still check the return so a rejected attribute errors loudly
      // instead of silently failing the subsequent launch.
      cudaError_t err = cudaFuncSetAttribute(
          staged_kernel<scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      TORCH_CHECK(err == cudaSuccess,
                  "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem,
                  ") failed: ", cudaGetErrorString(err));
    }
    staged_kernel<scalar_t><<<M, kBlock, smem, stream>>>(in, out, scales, N);
  } else {
    recompute_kernel<scalar_t><<<M, kBlock, 0, stream>>>(in, out, scales, N);
  }
}

void fused_silu_mul_quant_int8_launch(const at::Tensor& input,
                                      at::Tensor& out,
                                      at::Tensor& scales) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && scales.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(input.dtype() == at::kBFloat16 || input.dtype() == at::kHalf,
              "input must be bfloat16 or float16");
  TORCH_CHECK(out.dtype() == at::kChar, "out must be int8");
  TORCH_CHECK(scales.dtype() == at::kFloat, "scales must be float32");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, 2N]");
  TORCH_CHECK(input.size(1) % 2 == 0, "input last dim must be even");

  const int M = (int)input.size(0);
  const int N = (int)(input.size(1) / 2);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N, "out must be [M, N]");
  TORCH_CHECK(scales.numel() == M, "scales must have M elements");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(), "input/out must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

  if (M == 0 || N == 0) return;

  // Pin to the tensors' device so the cap query and the launch target the same GPU
  // (the current device is not guaranteed to be input.device() under TP/PP).
  const at::cuda::CUDAGuard device_guard(input.device());
  const int dev = input.get_device();

  // SMEM budget: stage when the dynamic y[] buffer plus the static/driver reserve fits the
  // opt-in cap; else fall back to recompute. Reserving kSmemReserve guarantees the dynamic
  // opt-in request in launch() can never crowd out the static s_warp_max[] (which would make
  // the staged launch fail), so everything above the threshold is covered by recompute_kernel.
  int cap = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  if (cap <= 0) cap = 48 * 1024;
  const size_t stage_cap = (cap > kSmemReserve) ? (size_t)(cap - kSmemReserve) : 0;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev);

  if (input.dtype() == at::kBFloat16) {
    size_t smem = (size_t)N * sizeof(__nv_bfloat16);
    bool stage = smem <= stage_cap && smem <= (size_t)INT_MAX;
    launch<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        out.data_ptr<int8_t>(), scales.data_ptr<float>(), M, N, smem, stage, stream);
  } else {
    size_t smem = (size_t)N * sizeof(__half);
    bool stage = smem <= stage_cap && smem <= (size_t)INT_MAX;
    launch<__half>(
        reinterpret_cast<const __half*>(input.data_ptr()),
        out.data_ptr<int8_t>(), scales.data_ptr<float>(), M, N, smem, stage, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(fused_silu_int8, m) {
  m.def("fused_silu_mul_quant_int8(Tensor input, Tensor! out, Tensor! scales) -> ()");
}

TORCH_LIBRARY_IMPL(fused_silu_int8, CUDA, m) {
  m.impl("fused_silu_mul_quant_int8", &fused_silu_mul_quant_int8_launch);
}
