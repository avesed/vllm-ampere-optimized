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

## Validated throughput

Apples-to-apples on **one RTX 3090**, Qwen3.5-9B, single-GPU (no tensor-parallel), 512-token prompt.
All numbers **tok/s**:

| engine · quant | decode | prefill (512-tok) | batch-16 |
|---|---:|---:|---:|
| stock vLLM · W4A8 | ❌ won't load | — | — |
| stock vLLM · W4A16 | 88 | 3.9k | 671 |
| this fork · W4A16 | 88 | 3.9k | 671 |
| **this fork · W4A8** | 87 | **5.8k** | **757** |

- The fork's **W4A16 is byte-identical to stock** — the patch only *adds* W4A8, zero regression elsewhere.
- **W4A8 vs W4A16: +46% prefill, +13% batched, decode parity** — and it's the only way to run W4A8 on Ampere.

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
