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
- **int8-act opt-in flag + MoE support** (`patches/0005`–`0006`) — `--marlin-input-dtype int8` (or env
  `VLLM_MARLIN_INPUT_DTYPE=int8`) turns a W4A16 checkpoint into W4A8 at serve time, for **dense and MoE**.

`vllm/` and `flashinfer/` carry the edits baked in; `patches/` + `scripts/revendor.sh` replay them on an
upstream bump, and `scripts/build_image_source.sh` builds + pushes the image.

## Results

Stock vLLM **won't load W4A8 on any Ampere GPU** — the fork is the only way to run it. Numbers below are
**W4A16 → W4A8** on the same fork engine, tok/s (int4 g32 AWQ+mse, cudagraph):

| GPU · arch | model | prefill (8k) | int8 Δ | decode | batch-32 |
|---|---|---|---:|---:|---:|
| RTX 3090 ×1 · `sm_86` | 9B dense | 4.7k → **7.0k** | **+49%** | 87 → 85 | 438 → **595** |
| RTX 3090 ×2 (pp2) · `sm_86` | 35B-A3B MoE | 9.1k → **10.8k** | **+19%** | 122 → 120 | 614 → **669** |
| A100 ×1 · `sm_80` | 9B dense | 11.1k → 11.1k | +0% | — | — |
| A100 ×1 · `sm_80` | 35B-A3B MoE | 22.8k → 24.7k | +8% | — | — |

- **The int8 prefill win is a consumer-`sm_86` effect** — those cards' fp16 tensor (FP32 accumulate) is
  half-rate, so int8 is a ~4× compute lever. A100 (`sm_80`) fp16 is full-rate → int8 prefill ~0 (dense)
  / +8% (MoE). The W4A8 enabler + int4-weight decode/VRAM savings hold on every Ampere card.
- **No-NVLink multi-GPU → `-pp 2 -tp 1`** — TP's all-reduce eats ~half of prefill (it shrinks the 35B
  int8 gain to +5%).
- **Quality:** decode is W4A16-parity; int8 activations cost ~zero accuracy — GSM8K (thinking) 9B W4A16
  81.6% / W4A8 85.6% (N=250); 35B-A3B W4A8 GSM8K 95.8%, MMLU-Pro 80.5%. The fork's W4A16 is byte-identical to stock.

## Use

```bash
docker run --gpus all -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model Avesed/Qwen3.6-27B-INT4-W4A16 --marlin-input-dtype int8 --pipeline-parallel-size 2 --max-model-len 8192
```
*(cu130 image needs NVIDIA driver ≥ 580.65. With NVLink use `--tensor-parallel-size 2`; single GPU, drop both.)*

Run a plain **W4A16** checkpoint as **W4A8** by adding **`--marlin-input-dtype int8`** (dense or MoE).

**No Docker?** A from-source wheel (`sm_80`+`sm_86`) is on
[Releases](https://github.com/avesed/vllm-ampere-optimized/releases) — `pip install` it (needs torch 2.11
+ CUDA 13). Enable W4A8 with `--marlin-input-dtype int8`.

- **Ready-made quants** — [huggingface.co/Avesed](https://huggingface.co/Avesed):
  - Qwen3.6-27B: [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-27B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 82.4%
  - Qwen3.6-35B-A3B (MoE): [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 80.2%
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>` — best quality = the AWQ + mse + g32 recipe in [`quantize/README.md`](quantize/README.md)

## Credits

Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0); W4A8 enablement follows upstream [#38066](https://github.com/vllm-project/vllm/pull/38066).
