# Benchmark results

**Hardware:** 2× NVIDIA RTX 3090 (24 GB, sm_86), no NVLink.
**Software:** vLLM v0.21-dev (tensor_parallel_size=2), CUDA graphs ON unless noted.
**Model:** Qwopus3.6-27B-v2 (Qwen3.5 hybrid-attention reasoning, 27B), abliterated.
**Method:** `benchmarks/vllm_verify.py <model> <tag>` — single-stream decode (batch 1),
batch-16 aggregate decode, and 8-prompt prefill (≈2 k tokens each).

## Headline: W4A8 wins (or ties) everywhere except on-disk size

| quant | kernel | single-stream decode | batch-16 decode | prefill | on-disk | VRAM weights |
|---|---|---:|---:|---:|---:|---:|
| W4A16 | Marlin WNA16 | 50.6 tok/s | 387 tok/s | 1045 tok/s | 27 G | int4 |
| W8A8  | CUTLASS int8 | 38.8 tok/s | 343 tok/s | **1250 tok/s** | 34 G | int8 |
| **W4A8** | **Marlin (patched)** | **50.4 tok/s** | **393 tok/s** | **1229 tok/s** | 34 G | **int4** |

> The 27B test model above: bf16 = 52 G. Only the FFN/attention `Linear` layers are quantized;
> embeddings, lm_head, the linear-attention/vision/MTP branches stay bf16 (~17 G), which is why
> none of the quants shrink to a pure-4-bit fraction.

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

## On-disk size ≠ VRAM (why W4A8 is 34 G on disk, same as W8A8)

W4A8 is **as big as W8A8 on disk** (34 G), and bigger than the plain int4 W4A16 (27 G). That is
**not** a quantization failure — it's the compressed-tensors storage format:

| | weight tensor on disk | bytes/weight | scales |
|---|---|---:|---|
| W4A16 (`pack-quantized`) | `int32`, 8 int4s packed per word | 0.5 | group-128 |
| W4A8 (`int-quantized`) | `int8`, **one byte per 4-bit value (not packed)** | 1.0 | group-128 |
| W8A8 (`int-quantized`) | `int8` | 1.0 | per-channel |

The `int-quantized` format that the W4A8Int kernel requires **does not bit-pack** — each 4-bit
weight (range −8..7) sits in a full `int8` byte, so on disk it costs the same as int8. W4A8 even
edges out W8A8 slightly because its group-128 scales (`[out, in/128]`) are larger than W8A8's
per-channel scales (`[out, 1]`).

**But VRAM is int4.** The Marlin patch repacks the on-disk `int8` into packed int4 (`(w+8)&0xF`,
8 per `int32`) at load time, so the GPU weight footprint — and decode bandwidth — is genuinely
half of int8. The proof is in the table: W4A8 decodes at 50.4 tok/s (≈ W4A16's 50.6), not at
W8A8's 38.8 — impossible if the kernel were reading int8 weights from VRAM.

Practical takeaway: W4A8's wins are all **runtime** (int4 decode bandwidth + int8 prefill + room
for CUDA graphs). It has **no disk/download advantage** — if you care about storage, ship W4A16.

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
