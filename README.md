# vllm-ampere-optimized

Automated rebuilds of **vLLM for the whole NVIDIA Ampere line** — A100 (`sm_80`) and
RTX 3090 / A40 / A6000 / A10 (`sm_86`) — plus the patches that unlock quantization paths vLLM
ships gated-to-Hopper or mis-configured for Ampere.

On every upstream vLLM release, GitHub Actions applies our patch series and builds, for
`TORCH_CUDA_ARCH_LIST="8.0 8.6"`:

- a **pip-installable wheel** → attached to a GitHub Release, and
- a **Docker image** → pushed to `ghcr.io`.

Two-arch (not the stock ~8-arch) → faster builds, smaller artifacts, nothing an Ampere card can't
use. The flagship patch enables **W4A8-INT8 on Ampere via Marlin** (vLLM gates its W4A8 kernels to
Hopper) — see below.

> Not a vendored fork: this repo holds only patches + CI + thin scripts. CI checks out upstream at
> the tag, patches it, builds. Why: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

```bash
# wheel (any Ampere host with matching torch/CUDA):
pip install https://github.com/<owner>/vllm-ampere-optimized/releases/download/v0.23.0-ampere/<wheel>.whl

# image (default cu130; needs NVIDIA driver >= 580.65.06 — use the cu129 tag on older drivers):
docker run --gpus all -p 8000:8000 \
  ghcr.io/<owner>/vllm-ampere-optimized:v0.23.0-ampere-cu130 \
  --model <hf-id> --max-model-len 8192
```

## Tip: 2+ consumer Ampere GPUs without NVLink → enable PCIe P2P

RTX 3090 (and other GeForce 30-series) have **no NVLink**, and the stock NVIDIA driver disables
GPU-to-GPU peer DMA over PCIe for product segmentation — the silicon supports it. On
tensor-parallel serving (`-tp 2`) that missing P2P is a real throughput tax.

The community [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules)
driver fork force-enables P2P over PCIe BAR1. With it, vLLM's two-shot **custom all-reduce turns on
for TP=2 automatically — no vLLM change** — recovering much of NVLink's ~50% TP gain (measured
**~10–30%** end-to-end on dual-3090 35B-A3B). **Strongly recommended for any no-NVLink 30-series
multi-GPU box.**

```bash
nvidia-smi topo -p2p r     # after patching: want "OK" between the GPUs, not "NS"
```

Caveats: only some 3090 SKUs expose a large/resizable **BAR1** (required) — verify first; needs
`iommu=pt` on the kernel cmdline; **bare-metal Linux only**; pins you to a forked driver branch. If
P2P won't come up, prefer **pipeline-parallel** (`-pp 2 -tp 1`) over TP=2 — far less PCIe-sensitive.

## How it works

Everything builds on **official upstream artifacts** — we never self-compile by default (for a
pure-Python patch a self-compile gives no inference speedup; upstream's image already carries sm_86
SASS, so single-arch buys only size/build-time).

```
watch-upstream.yml (cron 6h)  →  new release tag ≠ marker?  →  build.yml
   build.yml:  drift-check  →  { wheel (VLLM_USE_PRECOMPILED) → GH Release ,  overlay image → ghcr + smoke }  →  bump marker
```

- **Fast-path wheel** — `VLLM_USE_PRECOMPILED=1` reuses upstream's prebuilt kernels: a patched,
  installable wheel in minutes with **no CUDA compile** (runs on `ubuntu-latest`).
- **Overlay image (default)** — `FROM vllm/vllm-openai:<tag>` + our pure-Python patch + the device
  configs, **zero CUDA compile** → ~1–2 min on any runner; tagged `:<tag>` and `:latest`. Identical
  runtime to a self-compile for a pure-Python patch.
- **From-source single-arch image (opt-in)** — set repo var `BUILD_RUNNER` to a GPU runner (e.g. the
  2×3090 box) to *also* build an `8.0/8.6`-only fatbin via upstream `docker/Dockerfile`, tagged
  `:<tag>-ampere-cuXXX`. Only worth it for a smaller artifact, or once a patch touches native code
  (the `apply_patches.sh` guard forces this path then).
- **Drift canary** — `patch-drift-check.yml` runs `git apply --check` daily and opens an issue when
  a new upstream release shifts the code our patches anchor to.

Full setup + release flow: [`docs/RELEASE.md`](docs/RELEASE.md). Patch maintenance:
[`docs/PATCHING.md`](docs/PATCHING.md).

---

## Flagship patch: W4A8-INT8 on Ampere (`patches/0001-…`)

**W4A8** (int4 weights + int8 dynamic-per-token activations) is the **best serving quant on
Ampere** — int4 weight bandwidth for decode *and* int8 tensor-core compute for prefill. vLLM's
dedicated W4A8 kernels (`CutlassW4A8LinearKernel`, `MacheteLinearKernel`) require **Hopper sm_90**.
The **Marlin** kernel *can* do W4A8-int8 on Ampere — its CUDA side already accepts `is_a_8bit` — but
a Python config bug (`act_type=bf16` instead of `torch.int8`) made the layer fall through to
weight-only WNA16 → `Failed to find a kernel…`. Upstream
[vllm#38064](https://github.com/vllm-project/vllm/issues/38064) /
[#38066](https://github.com/vllm-project/vllm/pull/38066); our `0001-*.patch` carries the fix
(`act_type=torch.int8`; allow `int4` in the 8-bit-act assert; pack signed int4→uint4b8; add `int4`
to supported types).

Measured on 2× RTX 3090 (tp2, CUDA graphs on):

| quant (kernel) | single-stream decode | batch-16 decode | prefill |
|---|---|---|---|
| W4A16 (Marlin WNA16) | 50.6 tok/s | 387 tok/s | 1045 tok/s |
| W8A8 (CUTLASS int8) | 38.8 tok/s | 343 tok/s | **1250 tok/s** |
| **W4A8 (Marlin, patched)** | **50.4 tok/s** | **393 tok/s** | **1229 tok/s** |

W4A8 = W4A16's decode **and** W8A8's prefill, and small enough in VRAM to keep CUDA graphs (W8A8
OOMs graph capture at serving batch on 24 GB). Numbers + roofline interpretation + the
"int-quantized is int8-sized on disk" caveat: [`benchmarks/results.md`](benchmarks/results.md).

### Use the patch standalone (no rebuild)

Already running stock vLLM? Hot-patch the installed package in place (pure-Python, no recompile):

```bash
python patches/w4a8_int_marlin_ampere.py            # auto-detects the installed vllm; --revert to undo
```

### Quantize a model to W4A8

Use llm-compressor's `scheme="W4A8"` shortcut (saves `int-quantized`, which routes to the
W4A8Int/Marlin path — a manual `config_groups` saves `pack-quantized` → weight-only WNA16):

```bash
python quantize/quantize_w4a8.py <hf_model> <out_dir>
```

## Other patch: RTX 3090 fused-MoE configs (`configs/fused_moe/`)

vLLM ships fused-MoE Triton tile configs only for a handful of datacenter GPUs; on a 3090 it falls
back to a generic heuristic (`Using default MoE config. Performance might be sub-optimal!`).
`configs/fused_moe/*.json` carries RTX-3090-tuned configs, copied into the build by
`apply_patches.sh` — faster MoE forward, **zero accuracy change**, no kernel/native change (fast-path
stays valid). Currently: a 256-expert / N=512 `int4_w4a16` MoE. See
[`configs/README.md`](configs/README.md).

## Repo layout

```
patches/    0001-marlin-w4a8-int8-ampere.patch   # the load-bearing diff (CI source of truth)
            w4a8_int_marlin_ampere.py            # same fix as a standalone live hot-patch
            regenerate.py                        # re-emit the diff against a new upstream tag
.github/workflows/
            watch-upstream.yml  build.yml  patch-drift-check.yml
scripts/    apply_patches.sh  build_wheel_fastpath.sh  build_image_overlay.sh  build_image_ampere.sh  smoke_test.sh
docker/     Dockerfile.overlay  Dockerfile.wrapper  docker-bake.hcl
configs/    fused_moe/*.json                     # device-tuned MoE kernel configs (RTX 3090), copied in by apply_patches
quantize/   quantize_w4a8.py                     # llm-compressor recipe -> int-quantized W4A8
benchmarks/ vllm_verify.py  vllm_batch_sweep.py  results.md
docs/       ARCHITECTURE.md  PATCHING.md  RELEASE.md
UPSTREAM_VLLM_VERSION                            # last-built upstream tag (build marker)
```

## Credits

W4A8 Marlin enablement follows upstream
[vllm#38066](https://github.com/vllm-project/vllm/pull/38066). QQQ — an alternative Ampere W4A8
path — is [vllm#5218](https://github.com/vllm-project/vllm/pull/5218) /
[HandH1998/QQQ](https://github.com/HandH1998/QQQ). Built on
[vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0).
