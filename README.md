# vllm-ampere-optimized

A vendored, **built-from-source** Ampere fork of **vLLM v0.23.0** + **FlashInfer v0.6.12** for the
whole NVIDIA Ampere line — A100 (`sm_80`) and RTX 3090 / A40 / A6000 / A10 (`sm_86`). It un-gates
**W4A8 (int4 weights + int8 activations)** and adds native int8 kernels that upstream restricts to Hopper.

- **Image:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized) — built from the pinned vendored source on every change to `main`.

## What's in the fork

- **W4A8 on Ampere** (`patches/0001`, upstream [#38066](https://github.com/vllm-project/vllm/pull/38066)) —
  routes int4-weight + int8-activation through Marlin. The big serving win (table below). Pure-Python.
- **int8 8-row Marlin decode tile** (`patches/0002`, native `.cu`) — completes the W4A8 small-batch decode path.
- **int8-QK prefill attention** (`flashinfer/`, int8 IMMA in `compute_qk`) — int8 QK^T + fp16 PV for the
  full-attn layers of head_dim-256 hybrids (Qwen3.5/3.6), validated cos 0.9999 vs fp16; a long-context prefill lever.
- **AOT-compile cache-key fix** (`patches/0003`) — keys the torch.compile cache on the quant scheme
  (without it a W4A8 model loads a W4A16's cached graph and crashes — `KeyError: weight_zero_point`).

The int8 kernels are native (`.cu`/`.cuh`), so the fork is **vendored + built from source**: `vllm/`
(v0.23.0) and `flashinfer/` (v0.6.12) carry the edits baked in; `patches/` + `scripts/revendor.sh`
regenerate them on an upstream bump, and `build.yml` builds the image from source on every push to `main`.

## Why this exists

W4A8 is the best serving quant for Ampere — int4 weights cut decode bandwidth, int8 activations speed
up prefill — and **Marlin can run it on Ampere**. But vLLM gates its W4A8 path to Hopper: on an Ampere
GPU, loading a W4A8 checkpoint **crashes at load** (a weight-shape mismatch in
`process_weights_after_loading`), so Ampere users are stuck on W4A16. This fork routes W4A8 through
Marlin — plus the native int8 paths above — for the whole Ampere line.

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
