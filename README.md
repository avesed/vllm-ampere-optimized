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

## How it works

```
watch-upstream.yml (cron 6h)  →  new release tag ≠ marker?  →  build.yml
   build.yml:  drift-check  →  { wheel-fastpath → GH Release ,  image → ghcr + smoke test }  →  bump marker
```

- **Fast-path wheel** — our patches are pure-Python, so `VLLM_USE_PRECOMPILED=1` reuses upstream's
  prebuilt kernels: a patched, installable wheel in minutes with **no CUDA compile** (runs on
  `ubuntu-latest`). A native-code guard forces a from-source build if a patch ever touches `.cu`.
- **From-source image** — upstream `docker/Dockerfile --target vllm-openai`, driven entirely by
  `--build-arg torch_cuda_arch_list="8.0 8.6"`. Best on a self-hosted 2×3090 runner (set repo var
  `BUILD_RUNNER`); falls back to a GitHub-hosted runner with a disk-cleanup step.
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

## Repo layout

```
patches/    0001-marlin-w4a8-int8-ampere.patch   # the load-bearing diff (CI source of truth)
            w4a8_int_marlin_ampere.py            # same fix as a standalone live hot-patch
            regenerate.py                        # re-emit the diff against a new upstream tag
.github/workflows/
            watch-upstream.yml  build.yml  patch-drift-check.yml
scripts/    apply_patches.sh  build_wheel_fastpath.sh  build_image_ampere.sh  smoke_test.sh
docker/     Dockerfile.wrapper  docker-bake.hcl
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
