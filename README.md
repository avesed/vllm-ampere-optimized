# vllm-ampere-optimized

vLLM with **W4A8 (int4 weights + int8 activations) un-gated for the whole NVIDIA Ampere line** —
A100 (`sm_80`) and RTX 3090 / A40 / A6000 / A10 (`sm_86`). Shipped as a patch over the **official**
upstream build — pip wheel + Docker image, auto-rebuilt on every vLLM release. No fork to maintain,
no self-compile.

- **Wheels:** [Releases](https://github.com/avesed/vllm-ampere-optimized/releases) · **Images:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized)

## Why this exists

W4A8 is the best serving quant for Ampere — int4 weights cut decode bandwidth, int8 activations speed
up prefill — and **Marlin can run it on Ampere**. But vLLM gates its W4A8 path to Hopper: on an Ampere
GPU, loading a W4A8 checkpoint **crashes at load** (a weight-shape mismatch in
`process_weights_after_loading`), so Ampere users are stuck on W4A16.

This repo's patch (`patches/0001`, upstream [#38066](https://github.com/vllm-project/vllm/pull/38066))
routes W4A8 through Marlin so the whole Ampere line can run it — shipped automatically on top of each
official upstream release.

## Validated throughput & quality

Apples-to-apples on **one RTX 3090**, Qwen3.5-9B (int4 group-32, AWQ + mse quant), single-GPU,
cudagraph. All numbers **tok/s**:

| quant | decode (single) | prefill (8k prompt) | batch-32 |
|---|---:|---:|---:|
| stock vLLM · W4A8 | ❌ won't load on Ampere | — | — |
| this fork · W4A16 | 87 | 4.7k | 438 |
| **this fork · W4A8** | 85 | **7.0k** | **595** |

- **W4A8 vs W4A16: decode parity, +49% prefill, +36% batched.** int8 activations win the
  compute-bound regimes (prefill, large batch); int4 weights keep decode bandwidth-bound at parity.
  W4A8 is the better serving quant — and it's the only way to run W4A8 on Ampere at all.
- **Quality preserved:** GSM8K with thinking — W4A16 **81.6%** vs W4A8 **85.6%** (N=250). int8
  dynamic-per-token activations carry ~zero quality cost under the AWQ+mse weight quant.
- The fork's **W4A16 is byte-identical to stock** — the patch only *adds* W4A8, zero regression elsewhere.
- W4A8 + cudagraph also needed fixing a vLLM AOT-compile cache-key collision (`patches/0003`): the
  compile cache key omitted the quant scheme, so a W4A8 model could load a W4A16's cached graph and
  crash (`KeyError: weight_zero_point`). The fork keys the cache on the quant scheme.

Flagship 27B on 2× RTX 3090 (tp2): W4A8 ≈ 50 tok/s single-stream decode, 393 batched, 1229 prefill.

## Use

```bash
docker run --gpus all -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model Avesed/Qwen3.6-27B-W4A8 --tensor-parallel-size 2 --max-model-len 8192
```
*(default cu130 image needs NVIDIA driver ≥ 580.65)*

- **Ready-made W4A8:** [Avesed/Qwen3.6-27B-W4A8](https://huggingface.co/Avesed/Qwen3.6-27B-W4A8) (of official [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B))
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>`

## Credits

Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0); W4A8 enablement follows
upstream [#38066](https://github.com/vllm-project/vllm/pull/38066).
