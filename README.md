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
  --model Avesed/Qwen3.6-27B-W4A8 --tensor-parallel-size 2 --max-model-len 8192
```

`Avesed/Qwen3.6-27B-W4A8` is a ready-made W4A8 of the official **[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)** — serves ~47 tok/s single-user (sub-second TTFT), ~416 tok/s saturated on 2× RTX 3090. See [W4A8 below](#w4a8-on-ampere--the-flagship-patch).

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

- **Ready-made W4A8 models:** [Avesed/Qwen3.6-27B-W4A8](https://huggingface.co/Avesed/Qwen3.6-27B-W4A8) (W4A8 of official [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B))
- **Quantize your own:** `python quantize/quantize_w4a8.py <hf-model> <out-dir>`
- **Patch a running vLLM in place** (no rebuild): `python patches/w4a8_int_marlin_ampere.py`

## Other automatic Ampere fast paths (no flag, no patch — just know they exist)

Beyond the W4A8 patch, upstream vLLM already routes several quant paths to Ampere-friendly kernels
automatically — worth knowing so you pick the right checkpoint:

- **fp8 weight-only → W8A16-fp8 Marlin** (`<sm89`, i.e. all Ampere): an fp8 dense checkpoint auto-uses
  the Marlin fp8 fast path (~1.6× throughput, ~2× less weight VRAM vs bf16). Ampere has no fp8 tensor
  cores, but weight-only fp8 (dequant → bf16 compute) still wins on bandwidth. No flag needed.
- **AllSpark uint8b128 W8A16**: a uint8 W8A16 checkpoint (`group_size=-1`) gets an explicit Ampere branch
  that beats generic Marlin at small-M (low-batch) decode.
- **W8A8-int8 CUTLASS** (cap ≥ 7.5): int8 W8A8 checkpoints run the CUTLASS int8 path on Ampere — at the
  int8 tensor-core ceiling (great prefill; but decode reads int8 weights, so W4A8 wins at low batch).

## Hybrid / linear-attention models (GatedDeltaNet, Mamba2, …)

Modern hybrids (Qwen3.5/3.6, Jamba, Nemotron-H, …) run their linear-attention / SSM / causal-conv1d
layers on **vendored Triton kernels** shipped inside vLLM — **no separate `flash-linear-attention` /
`causal-conv1d` / `mamba-ssm` install is needed** (pinning them is a no-op). These JIT-Triton kernels
are not AOT-compiled, so this fork's CI runs the upstream mamba/GDN kernel test suite on a real Ampere
GPU (`scripts/ampere_kernel_ci.sh`, gated on `BUILD_RUNNER`) to catch sm_80/sm_86 codegen or numeric
regressions after each upstream/torch/triton bump — **622 kernel cases verified green on sm_86**. The
GatedDeltaNet recurrent-decode state is already bf16 by default (bandwidth-optimal) — nothing to tune.

## Multi-GPU without NVLink (30-series)

Consumer Ampere has no NVLink and the stock driver blocks PCIe P2P. The
[tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) driver fork
force-enables it → vLLM custom all-reduce turns on for `-tp 2` automatically (**~10–30%** on
dual-3090). Verify with `nvidia-smi topo -p2p r`; needs a large BAR1 + `iommu=pt`, bare-metal Linux.
If P2P won't come up, prefer `-pp 2 -tp 1`.

## How it works

`watch-upstream` (cron) sees a new release → `build` patches and ships, all from **official upstream
artifacts**:

- **wheel** = official upstream wheel + our pure-Python patch overlaid, repacked → GitHub Release.
  Reuses the official wheel's compiled `.so` (incl. `vllm/_moe_C.abi3.so`), so **MoE models work**.
  (The old `VLLM_USE_PRECOMPILED` fast-path fetched a stable-ABI `.so` subset that dropped `_moe_C`
  and silently broke every MoE model — fixed in `scripts/build_wheel_overlay.sh`.)
- **overlay image** `FROM vllm/vllm-openai:<tag>` + patch → ghcr `:latest` (also MoE-capable: it
  inherits the official image's full kernel set).
- from-source single-arch (`8.0/8.6`) image is opt-in (set repo var `BUILD_RUNNER`)

Setup + release flow: [docs/RELEASE.md](docs/RELEASE.md) · design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · patch maintenance: [docs/PATCHING.md](docs/PATCHING.md).

## Also included

- `configs/fused_moe/` — RTX 3090-tuned fused-MoE Triton configs (vLLM ships none for the 3090). Used by
  the **Triton** fused-MoE path (bf16/fp16/int8_w8a8); the Marlin `moe_wna16` W4A8/W4A16 path picks tiles
  in compiled CUDA and does not read these. Faster Triton MoE, zero accuracy change.
- `scripts/ampere_kernel_ci.sh` — runs the upstream mamba/GatedDeltaNet/causal-conv1d Triton kernel tests
  on a self-hosted Ampere GPU (`BUILD_RUNNER`) against the shipped image: anti-regression for the vendored
  JIT kernels (622 cases green on sm_86).
- `benchmarks/` — the decode/prefill serving harness behind the table above + an Ampere **diagnostic**
  harness (ncu IMMA-occupancy, comm-excluded decode-share, GDN kernel sweeps) for deciding whether a kernel
  lever is worth building. See [`benchmarks/README.md`](benchmarks/README.md) and [`results.md`](benchmarks/results.md).
- `docs/ROADMAP.md` — the prioritized shippable-optimization backlog (what's done / build-now / NO-GO).

## Credits

W4A8 Marlin enablement follows upstream [vllm#38066](https://github.com/vllm-project/vllm/pull/38066).
Built on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0).
