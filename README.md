# vllm-ampere-optimized

An Ampere fork of **vLLM v0.23.0** (+ FlashInfer v0.6.12) that un-gates **W4A8 (int4 weights + int8
activations)** and adds native int8 kernels upstream restricts to Hopper. Built from source for
`sm_80` (A100) and `sm_86` (RTX 3090 / A40 / A6000 / A10).

**Image:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized)

## Why

W4A8 is a strong serving quant — int4 weights cut decode bandwidth, int8 activations speed up prefill.
Marlin can run it on Ampere, but vLLM gates its W4A8 path to Hopper: on an Ampere GPU a W4A8 checkpoint
**crashes at load**, so Ampere users are stuck on W4A16. This fork routes W4A8 through Marlin so it runs.

## What's in it

- **W4A8 on Ampere** (`patches/0001`, upstream [#38066](https://github.com/vllm-project/vllm/pull/38066)) — int4-weight + int8-act through Marlin.
- **int8 8-row Marlin decode tile** (`patches/0002`) — completes the W4A8 small-batch decode path.
- **int8-QK prefill attention** (`flashinfer/`) — int8 QK^T + fp16 PV for head_dim-256 hybrids (Qwen3.5/3.6); a long-context prefill lever.
- **AOT-compile cache-key fix** (`patches/0003`) — keys the torch.compile cache on the quant scheme.

`vllm/` and `flashinfer/` carry the edits baked in; `patches/` + `scripts/revendor.sh` replay them on an
upstream bump, and `scripts/build_image_source.sh` builds + pushes the image.

## Throughput & quality

One RTX 3090, Qwen3.5-9B (int4 g32, AWQ + mse), single-GPU, cudagraph — tok/s:

| quant | decode | prefill (8k) | batch-32 |
|---|---:|---:|---:|
| stock vLLM · W4A8 | ❌ won't load on Ampere | — | — |
| this fork · W4A16 | 87 | 4.7k | 438 |
| **this fork · W4A8** | 85 | **7.0k** | **595** |

W4A8 vs W4A16: decode parity, **+49% prefill, +36% batch**. GSM8K (thinking): W4A16 81.6% vs W4A8 85.6%
(N=250) — int8 activations cost ~zero quality under AWQ+mse. The fork's W4A16 is byte-identical to stock.

**Hardware scope:** the int8 prefill/batch win is a **consumer-Ampere (`sm_86`)** effect — those cards'
fp16 tensor (FP32 accumulate) is half-rate, so int8 is a ~4× lever. On **A100 (`sm_80`)** fp16 is
full-rate, so the W4A8 *enabler* still applies but the int8 prefill speedup is small (~0 dense, +8% MoE).
int4-weight decode + VRAM savings hold on every Ampere card.

**Multi-GPU without NVLink → use `-pp 2 -tp 1`** (TP's per-layer all-reduce eats ~half of prefill). On
2×3090, pp2, 35B-A3B (MoE): W4A16 prefill-8k 9.1k → W4A8 **10.8k (+19%)**, decode 120, batch-32 669.

## Use

```bash
docker run --gpus all -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model Avesed/Qwen3.6-27B-W4A8 --pipeline-parallel-size 2 --max-model-len 8192
```
*(cu130 image needs NVIDIA driver ≥ 580.65. With NVLink use `--tensor-parallel-size 2`; single GPU, drop both.)*

**No Docker?** A from-source wheel (`sm_80`+`sm_86`) is on
[Releases](https://github.com/avesed/vllm-ampere-optimized/releases) — `pip install` it (needs torch 2.11
+ CUDA 13). Enable W4A8 with `--marlin-input-dtype int8`.

- **Ready-made W4A8:** [Avesed/Qwen3.6-27B-W4A8](https://huggingface.co/Avesed/Qwen3.6-27B-W4A8)
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>` — best quality = the AWQ + mse + g32 recipe in [`quantize/README.md`](quantize/README.md)

## Credits

Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0); W4A8 enablement follows upstream [#38066](https://github.com/vllm-project/vllm/pull/38066).
