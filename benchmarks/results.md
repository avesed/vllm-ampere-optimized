# Benchmark results

**Hardware:** 2× NVIDIA RTX 3090 (24 GB, sm_86), no NVLink.
**Software:** vLLM v0.21-dev (tensor_parallel_size=2), CUDA graphs ON unless noted.
**Model:** Qwopus3.6-27B-v2 (Qwen3.5 hybrid-attention reasoning, 27B), abliterated.
**Method:** `benchmarks/vllm_verify.py <model> <tag>` — single-stream decode (batch 1),
batch-16 aggregate decode, and 8-prompt prefill (≈2 k tokens each).

## Headline: W4A8 wins (or ties) everywhere except on-disk size

| quant | kernel | single-stream decode | batch-16 decode | prefill | size |
|---|---|---:|---:|---:|---:|
| W4A16 | Marlin WNA16 | 50.6 tok/s | 387 tok/s | 1045 tok/s | ~15 G |
| W8A8  | CUTLASS int8 | 38.8 tok/s | 343 tok/s | **1250 tok/s** | ~34 G |
| **W4A8** | **Marlin (patched)** | **50.4 tok/s** | **393 tok/s** | **1229 tok/s** | ~17 G |

## Reading the numbers

- **Decode is memory-bandwidth-bound** (each token reads every weight once; arithmetic
  intensity ≈ 2–4 FLOP/byte, far below the 3090's ~76 fp16 / ~150 int8 ridge). So the
  weight *byte count* dominates: int4 (W4A16, W4A8) reads half the bytes of int8 (W8A8)
  and decodes ~30 % faster. Activation precision is almost irrelevant here.
- **Prefill is compute-bound** (one big GEMM over the whole prompt; intensity well past the
  ridge). So int8 *tensor-core throughput* dominates: W8A8 and W4A8 both beat W4A16's fp16
  compute by ~18 %.
- **W4A8 = int4's decode + int8's prefill.** It reads int4 weight bytes (fast decode) and
  computes in int8 (fast prefill). That's why it matches or beats both single-scheme quants
  on every axis.
- **CUDA-graph headroom.** W4A8's small int4 weights leave enough VRAM to capture CUDA
  graphs at serving batch on 24 GB. **W8A8 OOMs graph capture** at the same batch and is
  forced to `enforce_eager`, which is what makes its decode worse than the raw int8 cost
  alone would predict. (See `vllm_batch_sweep.py` for the capture-size workaround that lets
  W8A8 graph-capture a limited set of batch sizes.)

## Notes on running W8A8 on Ampere

W8A8 works out of the box (`CutlassInt8ScaledMMLinearKernel`) — no patch needed — but it is
the wrong default on a 24 GB card: slower decode than int4 **and** can't keep CUDA graphs at
batch. Prefer it only for prefill-bound / large-batch throughput jobs, or on larger cards.

## Reproduce

```bash
# after applying patches/w4a8_int_marlin_ampere.py for the W4A8 row:
python benchmarks/vllm_verify.py /path/to/model-w4a16 w4a16
python benchmarks/vllm_verify.py /path/to/model-w8a8  w8a8
python benchmarks/vllm_verify.py /path/to/model-w4a8  w4a8
python benchmarks/vllm_batch_sweep.py /path/to/model-w4a8 w4a8   # 16/64/256 crossover
```

> Caveat: don't run with `VLLM_LOGGING_LEVEL=DEBUG` — it logs tensor `repr`s per op, forcing
> a GPU→CPU sync every rms_norm (~200× decode slowdown). The kernel-selection line is INFO.
