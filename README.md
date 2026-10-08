# vllm-ampere-optimized

An Ampere fork of **vLLM** (v0.31.0 + FlashInfer 0.6.18.post1; `UPSTREAM_VLLM_VERSION` and
`flashinfer/version.txt` track the vendored versions) that un-gates **W4A8 (int4 weights + int8
activations)** and adds native int8 kernels upstream restricts to Hopper. The fork's own kernels target
`sm_80` (A100) and `sm_86` (RTX 3090 / A40 / A6000 / A10). The image is a full multi-arch build, so
other GPUs run the stock vLLM paths.

**Image:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized)

## Why

W4A8 is a strong serving quant — int4 weights cut decode bandwidth, int8 activations speed up prefill.
Marlin can run it on Ampere, but vLLM gates its W4A8 path to Hopper: on an Ampere GPU a W4A8 checkpoint
**crashes at load**, so Ampere users are stuck on W4A16. This fork routes W4A8 through Marlin so it runs.

## What's in it

- **W4A8 on Ampere** (`patches/0001`, upstream [#38066](https://github.com/vllm-project/vllm/pull/38066)) — int4-weight + int8-act through Marlin.
- **int8 8-row Marlin decode tile** (`flashampere/marlin/`) — completes the W4A8 small-batch decode path.
- **AOT-compile cache-key fix** (`patches/0003`) — keys the torch.compile cache on the quant scheme.
- **int8-act opt-in flag + MoE support** (`patches/0005`–`0006`) — `--marlin-input-dtype int8` (or env
  `VLLM_MARLIN_INPUT_DTYPE=int8`) turns a W4A16 checkpoint into W4A8 at serve time, for **dense and MoE**.
- **Vendored famp Marlin kernel** (`flashampere/marlin/`) — the fork owns its W4A8/W4A16 Marlin GEMM
  as a plugin-selected `.so` (bit-exact vs stock, built in the image for `sm_80`+`sm_86`), so the int8
  path survives upstream refactors.
- **flashampere attention backend** (`flashampere/`, opt-in `VLLM_FLASHAMPERE=1`) — composable Ampere
  attention legs for head_dim ≤ 256: fp16-accumulate PV prefill (GeForce RTX-30 only; +1–4%
  long-context TTFT) and an opt-in XQA spec-verify kernel (`VLLM_FLASHAMPERE_XQA_VERIFY=1`; engages only
  when the KV page is ≤ 256 tokens, so not on the Qwen3.5/3.6 hybrids). Any case a leg does not cover
  falls back to the stock backend bit-for-bit. Larger heads (Gemma-4's head_dim 512) go to stock
  FlashInfer/Triton, which are now correct and faster there.
- **DSpark speculative decoding** — `--speculative-config '{"method":"dspark",...}'` serves DeepSeek
  DSpark block-diffusion draft heads (fork-only). DFlash is upstream too; the fork also serves DFlash
  heads that mix sliding and full attention on the V1 model runner, which upstream 0.31 rejects there.
  Ready-made head:
  [Avesed/Qwen3.6-27B-DSpark](https://huggingface.co/Avesed/Qwen3.6-27B-DSpark).
- **Capture-safe FA2 paged decode** — the image builds vllm-flash-attention from
  [avesed/flash-attention](https://github.com/avesed/flash-attention/tree/vllm-ampere-optimized/capture-safe-kvcache):
  the upstream pin's `mha_fwd_kvcache` does a device-to-host sync on paged KV, which breaks CUDA graph
  capture of the fork's FA2 spec-verify path; the fork skips that check while a stream is capturing.

`vllm/` and `flashinfer/` carry the edits baked in — the vendored tree *is* the fork, and nothing
applies `patches/` (it is the written record of the edits). `scripts/revendor.sh` 3-way merges the tree
onto a new upstream tag, and `scripts/build_image_source.sh` builds + pushes the image.

## Results

Stock vLLM **won't load W4A8 on any Ampere GPU** — the fork is the only way to run it. Numbers below are
**W4A16 → W4A8** on the same fork engine, tok/s (int4 g32 AWQ+mse, cudagraph). They were measured on
the vLLM 0.23-based releases (v0.2/v0.3) and have not been re-measured on 0.31:

| GPU · arch | model | prefill (8k) | int8 Δ | decode | batch-32 |
|---|---|---|---:|---:|---:|
| RTX 3090 ×1 · `sm_86` | 9B dense | 4.7k → **7.0k** | **+49%** | 87 → 85 | 438 → **595** |
| RTX 3090 ×2 (pp2) · `sm_86` | 35B-A3B MoE | 9.1k → **10.8k** | **+19%** | 122 → 120 | 614 → **669** |
| A100 ×1 · `sm_80` | 9B dense | 11.1k → 11.1k | +0% | — | — |
| A100 ×1 · `sm_80` | 35B-A3B MoE | 22.8k → 24.7k | +8% | — | — |
| RTX 3090 ×1 · `sm_86` | 26B-A4B dLLM | 4.2k → **5.5k** | **+32%** | 178 → **213** | — |

- **The int8 prefill win is a consumer-`sm_86` effect** — those cards' fp16 tensor (FP32 accumulate) is
  half-rate, so int8 is a ~4× compute lever. A100 (`sm_80`) fp16 is full-rate → int8 prefill ~0 (dense)
  / +8% (MoE). The W4A8 enabler + int4-weight decode/VRAM savings hold on every Ampere card.
- **No-NVLink multi-GPU → `-pp 2 -tp 1`** — TP's all-reduce eats ~half of prefill (it shrinks the 35B
  int8 gain to +5%).
- **Quality:** decode is W4A16-parity; int8 activations cost ~zero accuracy — GSM8K (thinking) 9B W4A16
  81.6% / W4A8 85.6% (N=250); 35B-A3B W4A8 GSM8K 95.8%, MMLU-Pro 80.5%. The fork's W4A16 is byte-identical to stock.
- **DiffusionGemma (block-diffusion dLLM, the row above):** prefill (8k) + single-stream gen, canvas 256. Unlike
  AR decode (≈flat under int8), the dLLM's generation is compute-bound too (every denoise step is prefill-like), so
  it gains as well (178 → 213, **+20%**). int8-act also beats **NVFP4 (176)** — no native FP4 kernel below `sm_89`,
  so it dequant-emulates to bf16 (memory-only, ≈ W4A16 speed) — at matched accuracy: W4A8 GSM8K 95.7% / MMLU-Pro
  76.4%; NVFP4 96.0% / 77.8% (N=1000/500). Two serve
  requirements: **`--dtype bfloat16`** (under int8, Gemma's large activations overflow fp16 — down-proj GEMM
  dequant > 65504 → inf → NaN → blank output) and **`--attention-backend TRITON_ATTN`** (the diffusion mixed
  causal/bidirectional mask isn't supported by flash-attn (wants FA4) or FlashInfer — the int8 win is the
  MoE GEMM alone). Run the
  [cyankiwi W4A16 ckpt](https://huggingface.co/cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4) with
  `VLLM_MARLIN_INPUT_DTYPE=int8 --dtype bfloat16 --attention-backend TRITON_ATTN` over the chat endpoint.

## Use

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model Avesed/Qwen3.6-27B-INT4-W4A16 --marlin-input-dtype int8 --pipeline-parallel-size 2 --max-model-len 8192
```
*(cu130 image needs NVIDIA driver ≥ 580.65. Keep `--ipc=host` (or `--shm-size=8g`): Docker's default
64 MB `/dev/shm` corrupts vLLM's multi-GPU input broadcast and `-tp 2` output turns to garbage. With NVLink use `--tensor-parallel-size 2`; single GPU, drop both. On a **no-NVLink** multi-GPU box, prefer `-pp 2`, or add `--disable-custom-all-reduce` if you use `-tp 2` — custom all-reduce over PCIe + `expandable_segments` can crash at startup.)*

Run a plain **W4A16** checkpoint as **W4A8** by adding **`--marlin-input-dtype int8`** (dense or MoE).

**Images and video** work on every fork path: W4A8 (dense and MoE), flashampere, and MTP / DFlash /
DSpark spec decode. The quant recipes, and the Qwen3.6 quants below, keep the vision tower in bf16.
The release matrix covers this in `scripts/release-matrix/t12_mm.sh`.

**No Docker?** Releases since v0.4 ship as the image only. The last wheel is on
[v0.3](https://github.com/avesed/vllm-ampere-optimized/releases/tag/v0.3) (vLLM 0.23, `sm_80`+`sm_86`,
torch 2.11 + CUDA 13) and predates everything after it.

- **Ready-made quants** — [huggingface.co/Avesed](https://huggingface.co/Avesed):
  - Qwen3.6-27B: [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-27B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 82.4%
  - Qwen3.6-35B-A3B (MoE): [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 80.2%
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>` — best quality = the AWQ + mse + g32 recipe in [`quantize/README.md`](quantize/README.md)

## Credits

Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0); W4A8 enablement follows upstream [#38066](https://github.com/vllm-project/vllm/pull/38066).
