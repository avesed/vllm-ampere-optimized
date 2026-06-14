# vllm-ampere-optimized

vLLM rebuilt for the whole **NVIDIA Ampere** line — A100 (`sm_80`) and RTX 3090 / A40 / A6000 / A10
(`sm_86`). On every upstream vLLM release, GitHub Actions applies our patch series to the
**official** upstream build and publishes a pip wheel + a Docker image. Headline patch: **W4A8-INT8
on Ampere via Marlin** (vLLM gates its W4A8 kernels to Hopper).

> Not a fork — just patches + CI. CI takes upstream at the release tag, patches it, ships it. No
> self-compile by default (for a pure-Python patch it gives no speedup; upstream already ships sm_86 SASS).

- **Repo:** https://github.com/avesed/vllm-ampere-optimized
- **Wheels:** [Releases](https://github.com/avesed/vllm-ampere-optimized/releases) · **Images:** [`ghcr.io/avesed/vllm-ampere-optimized`](https://github.com/avesed/vllm-ampere-optimized/pkgs/container/vllm-ampere-optimized)

## Install

```bash
# wheel — copy the .whl URL from the latest Release:
pip install https://github.com/avesed/vllm-ampere-optimized/releases/...

# image (default cu130 → needs NVIDIA driver ≥ 580.65.06):
docker run --gpus all -p 8000:8000 \
  ghcr.io/avesed/vllm-ampere-optimized:latest \
  --model <hf-id> --max-model-len 8192
```

## W4A8 on Ampere — the flagship patch

W4A8 (int4 weights + int8 activations) is the best serving quant on Ampere: int4 decode bandwidth +
int8 prefill compute. vLLM's dedicated W4A8 kernels are Hopper-only; **Marlin** can do it on Ampere,
but a config bug (`act_type=bf16` instead of `int8`) made it fall back to weight-only WNA16.
`patches/0001-marlin-w4a8-int8-ampere.patch` fixes it (upstream
[#38064](https://github.com/vllm-project/vllm/issues/38064) /
[#38066](https://github.com/vllm-project/vllm/pull/38066)). Measured on 2× RTX 3090 (tp2, CUDA graphs):

| quant | single-stream decode | batch-16 decode | prefill |
|---|---:|---:|---:|
| W4A16 | 50.6 tok/s | 387 tok/s | 1045 tok/s |
| W8A8  | 38.8 tok/s | 343 tok/s | **1250 tok/s** |
| **W4A8** | **50.4 tok/s** | **393 tok/s** | **1229 tok/s** |

- **Ready-made W4A8 model:** [Avesed/Qwopus3.6-27B-v2-abliterated-int4-w4a8](https://huggingface.co/Avesed/Qwopus3.6-27B-v2-abliterated-int4-w4a8)
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>`
- **Patch a running vLLM in place** (no rebuild): `python patches/w4a8_int_marlin_ampere.py`

## Multi-GPU without NVLink (30-series)

Consumer Ampere has no NVLink and the stock driver blocks PCIe P2P. The
[tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) driver fork
force-enables it → vLLM custom all-reduce turns on for `-tp 2` automatically (**~10–30%** on
dual-3090). Verify with `nvidia-smi topo -p2p r`; needs a large BAR1 + `iommu=pt`, bare-metal Linux.
If P2P won't come up, prefer `-pp 2 -tp 1`.

## How it works

`watch-upstream` (cron) sees a new release → `build` patches and ships, all from **official upstream
artifacts**:

- **wheel** via `VLLM_USE_PRECOMPILED` → GitHub Release
- **overlay image** `FROM vllm/vllm-openai:<tag>` + patch → ghcr `:latest`
- from-source single-arch (`8.0/8.6`) image is opt-in (set repo var `BUILD_RUNNER`)

Setup + release flow: [docs/RELEASE.md](docs/RELEASE.md) · design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · patch maintenance: [docs/PATCHING.md](docs/PATCHING.md).

## Also included

- `configs/fused_moe/` — RTX 3090-tuned fused-MoE Triton configs (vLLM ships none for the 3090) →
  faster MoE, zero accuracy change.
- `benchmarks/` — the decode/prefill harness behind the table above (+ interpretation in
  [`results.md`](benchmarks/results.md)).

## Credits

W4A8 Marlin enablement follows upstream [vllm#38066](https://github.com/vllm-project/vllm/pull/38066).
Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0).
