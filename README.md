# vllm-ampere-optimized

An Ampere fork of **vLLM** (v0.34.0, Model Runner V2; `UPSTREAM_VLLM_VERSION` tracks the vendored version),
shipped as a multi-arch image built from source. It targets `sm_80` (A100) and `sm_86` (RTX 3090 / A40 /
A6000 / A10); other GPUs run the stock vLLM paths.

Upstream vLLM now covers most of what this fork used to add on Ampere — W4A8 through Humming, DFlash and
DSpark drafters, the speculative-decoding correctness fixes. What the fork keeps is what upstream still
lacks, chiefly a much faster speculative-decoding verify at long context, plus measured serving recipes for
RTX 3090-class machines.

**Image:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized)

## What's in it

- **Split-KV spec-decode verify on FA2.** FA2's varlen kernel only splits the KV walk when the query is one
  token, so every MTP / DFlash / DSpark verify step (K+1 tokens) runs on a handful of SMs at long context.
  Uniform verify batches are routed to FA2 `fwd_kvcache`, which does split KV. On Qwen3.6-27B with MTP K=2
  this takes 16k-context decode from **61 to 95 tok/s**. `VLLM_FA2_KVCACHE_VERIFY=0` turns it off. Upstream is
  fixing the same gap inside the kernel
  ([flash-attention#176](https://github.com/vllm-project/flash-attention/pull/176),
  [vllm#51320](https://github.com/vllm-project/vllm/pull/51320)).
- **Capture-safe FA2 paged decode.** vllm-flash-attention is built from
  [avesed/flash-attention](https://github.com/avesed/flash-attention/tree/vllm-ampere-optimized/capture-safe-kvcache-034):
  the pinned `mha_fwd_kvcache` checks paged-KV lengths with a device-to-host sync, which breaks CUDA-graph
  capture of the route above; the fork runs that check on the device.
- **Fixes carried until upstream merges them:** Humming's fallback crash
  ([#68194](https://github.com/vllm-project/vllm/issues/68194); Qwen3.6-35B-A3B W4A8 does not load without it),
  sampling-parameter size limits ([#55164](https://github.com/vllm-project/vllm/pull/55164); an oversized
  `logit_bias`, `allowed_token_ids` or stop-token list used to kill the engine), a torch.compile cache key that
  includes the quantization scheme, int8 W8A8 MoE config selection with tuned RTX 3090 tables, and a
  FlexAttention packed-KV fix.
- **A release gate.** Every image passes `scripts/release-matrix/` on 2× RTX 3090 before it is published.

`vllm/` carries the edits baked in — the vendored tree *is* the fork, and nothing applies `patches/` (it is the
written record of past edits). `scripts/revendor.sh` 3-way merges the tree onto a new upstream tag, and
`scripts/build_image_source.sh` builds and pushes the image.

## Results

2× RTX 3090 (NVLink), TP2, Qwen3.6-27B W4A16 (AWQ, g32), all on one day at the same GPU power limits; tok/s,
medians of 5. Decode is streamed at ~200 and ~16k tokens of context; prefill is an 8k prompt over its TTFT.

| MTP K=2 | stock vLLM 0.34 | this image |
|---|---:|---:|
| decode, ctx ~200 | 120.2 | 124.0 |
| decode, ctx ~16k | 61.3 | **95.4** |
| no spec decode / 8k prefill | 67.0 / 2131 | 66.6 / 2127 |

Recommended configurations on this image (prefix caching on):

| configuration | decode ~200 | decode ~16k | prefill 8k |
|---|---:|---:|---:|
| W4A16 + MTP K=2 | 121.8 | 94.9 | 2074 |
| + draft-only NVFP4 `lm_head` | 137.4 | 103.4 | 2083 |
| + custom all-reduce (NVLink) | **149.7** | **110.8** | 2087 |
| + W4A8 through Humming | 149.3 | 118.1 | **3612** |
| W4A16 fp16 + fp16 accumulate, MTP, NVFP4 draft head, custom AR | **153.5** | 115.4 | 3134 |

- **Other models** (custom all-reduce on): Qwen3.6-35B-A3B W4A8 decode 184 tok/s, 8k prefill 13.5k tok/s
  (prefix caching off). DFlash K=7 on the 27B: 186 / 127 tok/s.
- **Accuracy:** GSM8K (sampled, 40 questions) 87.5–97.5% across the rows above, inside the run-to-run band of
  that sample size. Speculative decoding is lossless; quantized activations and fp16 accumulation change
  numerics, so measure your own workload before switching.
- **Limits of these numbers:** they compare within a table only — a different power limit or clock moves all of
  them. A100 was not re-measured for this release.

## Use

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model Avesed/Qwen3.6-27B-INT4-W4A16 --tensor-parallel-size 2 --max-model-len 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_lm_head_quantization":"nvfp4"}'
```
*(cu130 image needs NVIDIA driver ≥ 580.65. Keep `--ipc=host` (or `--shm-size=8g`): Docker's default 64 MB
`/dev/shm` corrupts vLLM's multi-GPU input broadcast and `-tp 2` output turns to garbage.)*

- **Two GPUs:** with an NVLink bridge, leave vLLM's custom all-reduce on (its default): +7–10% per decode step
  over NCCL here. Without NVLink, add `--disable-custom-all-reduce` or use `--pipeline-parallel-size 2`.
- **W4A8 (int8 activations) on any W4A16 compressed-tensors checkpoint, dense or MoE:**
  `--quantization humming` plus `-e VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8","group_size":0}'`.
  Prefill gains the most on GeForce cards (fp16 tensor cores with fp32 accumulation run at half rate there).
- **fp16 accumulation (W4A16 only):** `--quantization humming --dtype half -e VLLM_HUMMING_USE_F16_ACCUM=1`.
  Humming accepts it only for fp16 activations; it doubles peak MMA throughput on GA10x at the cost of
  precision (27B 8k prefill 2225 → 3193 tok/s here).
- **Speculative decoding:** MTP K=2 is the default recommendation. DFlash and DSpark heads load as published.
  DSpark accepted almost nothing on long, off-distribution prompts in our runs (stock vLLM behaves the same),
  so check it on your traffic.
- **Hybrid models and prefix caching:** on Qwen3.5-family models prefix caching splits prefill chunks to save
  recurrent state; on the 35B-A3B that cost 29% of cold 8k prefill. `--no-enable-prefix-caching` wins when
  prompts rarely repeat.

**Upgrading from v0.4.x:** `--marlin-input-dtype` and `VLLM_MARLIN_INPUT_DTYPE` are gone (upstream removed
Marlin W4A8; vLLM refuses to start with the variable set) — use the Humming flags above.
`VLLM_USE_V2_MODEL_RUNNER` must be unset or `1`. The flashampere backend and `VLLM_FLASHAMPERE*` are retired.

**Images and video** go through upstream vLLM paths; the fork's verify route only takes uniform text verify
batches. The quant recipes below keep the vision tower in bf16.

**No Docker?** Releases since v0.4 ship as the image only. The last wheel is on
[v0.3](https://github.com/avesed/vllm-ampere-optimized/releases/tag/v0.3) (vLLM 0.23) and predates everything
after it.

- **Ready-made quants** — [huggingface.co/Avesed](https://huggingface.co/Avesed):
  - Qwen3.6-27B: [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-27B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 82.4%
  - Qwen3.6-35B-A3B (MoE): [INT4-W4A16](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT4-W4A16) · [INT8-W8A8](https://huggingface.co/Avesed/Qwen3.6-35B-A3B-INT8-W8A8) — int4: GSM8K 96.8% / MMLU-Pro 80.2%
  - DSpark draft head: [Avesed/Qwen3.6-27B-DSpark](https://huggingface.co/Avesed/Qwen3.6-27B-DSpark)
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>` — best quality = the AWQ + mse + g32 recipe in [`quantize/README.md`](quantize/README.md)

## Credits

Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) and
[humming](https://github.com/vllm-project/humming); W4A8 on Ampere started from upstream
[#38066](https://github.com/vllm-project/vllm/pull/38066).
