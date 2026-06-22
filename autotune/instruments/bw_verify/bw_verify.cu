// bw_verify — bandwidth-saturating GDDR memory probe WITH integrity check, in one pass.
//
// Writes an index-derived (recomputable, no golden buffer) pattern across a large device
// buffer at peak bandwidth, then reads it back, recomputes the expected value, and counts
// mismatches. Reports achieved write/read GB/s + mismatch_count as JSON.
//
//   - read_GB_s    : the EDR-knee signal for a mem-OC sweep (rolls over past the knee) and the
//                    real achievable bandwidth at the current clock (no root needed).
//   - mismatch_count: a no-root, no-vLLM integrity signal — catches a no-ECC cell flip that the
//                    EDR knee can't see (>0 == corruption). NOTE: covers RAW cells only, not the
//                    model compute path — the inference golden token-id check is the real gate.
//
// Build:  nvcc -O3 -arch=sm_86 bw_verify.cu -o bw_verify
// Run:    ./bw_verify [size_gb=2] [iters=5]   (scope the GPU via CUDA_VISIBLE_DEVICES=GPU-<uuid>)
// No root. Pure CUDA compute access (works in an unprivileged container too).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ uint32_t pat(uint64_t i) {
    // cheap recomputable pattern (Knuth multiplicative hash) — no golden buffer needed
    return (uint32_t)((i * 2654435761ULL) ^ 0x9E3779B9ULL);
}

__global__ void k_write(uint32_t* buf, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        buf[i] = pat(i);
}

__global__ void k_read_verify(const uint32_t* buf, uint64_t n,
                              unsigned long long* mismatch, unsigned long long* checksum) {
    unsigned long long local_mm = 0, local_cs = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t v = buf[i];
        local_cs += v;
        if (v != pat(i)) local_mm++;
    }
    atomicAdd(mismatch, local_mm);
    atomicAdd(checksum, local_cs);
}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"{\"error\":\"%s at %d: %s\"}\n",#call,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static double median(double* a, int n) {
    for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) if (a[j]<a[i]){double t=a[i];a[i]=a[j];a[j]=t;}
    return n? a[n/2] : 0.0;
}

int main(int argc, char** argv) {
    double size_gb = (argc > 1) ? atof(argv[1]) : 2.0;
    int iters      = (argc > 2) ? atoi(argv[2]) : 5;
    if (iters < 1) iters = 1;

    CK(cudaSetDevice(0));  // caller scopes with CUDA_VISIBLE_DEVICES=GPU-<uuid>
    uint64_t bytes = (uint64_t)(size_gb * (1ull<<30));
    bytes &= ~((uint64_t)4095);                 // align
    uint64_t n = bytes / sizeof(uint32_t);
    if (n == 0) { fprintf(stderr,"{\"error\":\"size too small\"}\n"); return 2; }

    uint32_t* buf = nullptr;
    if (cudaMalloc(&buf, n * sizeof(uint32_t)) != cudaSuccess) {
        fprintf(stderr,"{\"error\":\"cudaMalloc %.2f GB failed\"}\n", size_gb); return 2;
    }
    unsigned long long *d_mm=nullptr, *d_cs=nullptr;
    CK(cudaMalloc(&d_mm, sizeof(unsigned long long)));
    CK(cudaMalloc(&d_cs, sizeof(unsigned long long)));

    int threads = 256, blocks = 0;
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, 0);
    blocks = (blocks ? blocks : 80) * 32;       // oversubscribe SMs to saturate DRAM

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double wr[64], rd[64]; if (iters>64) iters=64;
    unsigned long long total_mm = 0;

    for (int it=0; it<iters; it++) {
        // write pass
        CK(cudaEventRecord(a));
        k_write<<<blocks, threads>>>(buf, n);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms_w=0; CK(cudaEventElapsedTime(&ms_w, a, b));
        wr[it] = (double)bytes / (ms_w/1e3) / 1e9;          // GB/s (10^9)

        // read+verify pass
        CK(cudaMemset(d_mm, 0, sizeof(unsigned long long)));
        CK(cudaMemset(d_cs, 0, sizeof(unsigned long long)));
        CK(cudaEventRecord(a));
        k_read_verify<<<blocks, threads>>>(buf, n, d_mm, d_cs);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms_r=0; CK(cudaEventElapsedTime(&ms_r, a, b));
        rd[it] = (double)bytes / (ms_r/1e3) / 1e9;
        unsigned long long mm=0; CK(cudaMemcpy(&mm, d_mm, sizeof(mm), cudaMemcpyDeviceToHost));
        if (mm > total_mm) total_mm = mm;
    }

    printf("{\"read_GB_s\": %.1f, \"write_GB_s\": %.1f, \"mismatch_count\": %llu, "
           "\"bytes\": %llu, \"iters\": %d}\n",
           median(rd, iters), median(wr, iters), total_mm, (unsigned long long)bytes, iters);
    cudaFree(buf); cudaFree(d_mm); cudaFree(d_cs);
    return 0;
}
