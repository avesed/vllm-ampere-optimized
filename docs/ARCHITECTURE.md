# Architecture

The maintainer's mental model. Why this repo is shaped the way it is.

## Vendored fork, built from source

This repo vendors the **complete** modified source — `vllm/` (upstream v0.23.0) and `flashinfer/`
(v0.6.12) — with every edit baked in. It is **not** a patch-overlay, because the int8 work is
**native**: `.cu`/`.cuh` kernels (the int8 8-row Marlin decode tile in patch 0002, the int8-QK
FlashInfer IMMA path) that a pip-overlay onto an official wheel/image physically cannot carry. Native
code ships only from a real source build — which is exactly why the source is vendored rather than
patched at build time.

`patches/` is no longer applied during a build — it is the **recipe** (`regenerate.py` for 0001 +
`git apply` for 0002/0003 + `flashinfer_int8/apply_to_source.py`) that regenerates the vendored trees
from a fresh upstream checkout, so an upstream bump stays reproducible and drift is detectable
(`scripts/revendor.sh`; the github-hosted `patch-drift-check` canary replays it daily). See
`patches/README.md`.

## One build path: local, from source

The image is built **from the vendored source on a local GPU box** (`scripts/build_image_source.sh`:
vLLM → sm_80+sm_86 fatbin, then the int8-QK FlashInfer overlaid) and pushed to ghcr **by the
maintainer**. There is **no CI build**: a from-source vLLM CUDA build needs a GPU, and a self-hosted
GPU runner on a **public** repo is a security risk — a malicious PR could run arbitrary code on the
runner. The only CI is two github-hosted canaries (`watch-upstream`, `patch-drift-check`) that just
open issues; they never build or push. See `docs/RELEASE.md`.

## Arch: `TORCH_CUDA_ARCH_LIST="8.0 8.6"` (all Ampere)

A100 = `sm_80`, RTX 3090 / A40 / A6000 / A10 = `sm_86`. No `+PTX` — vLLM strips top-level `+PTX`
during CMake arch filtering, and every target device *is* one of these two arches, so JIT
forward-compat is dead weight.

**Hopper/Blackwell kernels skip cleanly, by design.** vLLM computes per-kernel SRC arch lists and
intersects each against the target arches via `cuda_archs_loose_intersection`. With only
`{8.0, 8.6}`, every sm_90/sm_100 block's arch var resolves to the empty list, so its
`if(... AND <ARCHS>)` is false and the block is skipped with a benign `STATUS` line. There is no
top-level assertion requiring sm_90 — an Ampere-only build is a supported configuration.

- **Skipped on Ampere:** Machete (`9.0a`), CUTLASS scaled_mm C3X sm90 int8/fp8, **CUTLASS W4A8
  sm90** (Hopper-only — exactly why we need the Marlin route), CUTLASS MoE grouped_mm sm90,
  FP4/scaled_mm sm100 (Blackwell), FlashMLA (`9.0a`, creates an empty target so setup.py still
  resolves — do not "fix" the missing `.so`).
- **Built for Ampere (our route is live):** Marlin fp16/bf16 (`8.0+PTX` → covers 8.6), Marlin
  "other" (`7.5;8.0+PTX`), CUTLASS scaled_mm **C2X int8 W8A8** (`7.5;8.0;8.7;8.9+PTX`). Marlin
  **FP8-input** needs `8.9` and is *not* on Ampere — fine, our patch is W4A8-**int8**.

**CI must fail only on** `CMake Error` / `error:` / `nvcc fatal`. The FlashMLA / scaled_mm_c3x_sm90
/ Machete / W4A8-sm90 / FP4-sm100 `STATUS` skips are expected.

## CUDA base: cu130 default, cu129 broad-compat

For Ampere, CUDA version is **not** a performance lever — the kernels are mature and compile
equivalently across CUDA 12.4–13.0; new-CUDA optimizations target Hopper/Blackwell. The choice is
driver floor + ecosystem:

- **cu130** (CUDA 13.0): upstream Dockerfile + `requirements/cuda.txt` (`nvidia-cutlass-dsl[cu13]`)
  default → from-source builds with **zero override**. Driver ≥ 580.65.06. Default here.
- **cu129** (CUDA 12.9): vLLM's pip-install default; driver ≥ 575. Best for shipping to arbitrary
  Ampere hosts. Produce it locally with `CUDA_VERSION=12.9.1 scripts/build_image_source.sh`.
- Downgrading to cu129 for the *source* build can trip the `nvidia-cutlass-dsl[cu13]` pin at some
  tags — verify it resolves, or stay on cu130.

## Caching

`sccache` → GitHub Actions cache backend (`mozilla-actions/sccache-action@v0.0.10`, `version:
v0.10.0` — older sccache silently fails the post-2025-02 cache service v2). GHA cache is 10 GB/repo
LRU; the two-arch object set fits where a full multi-arch set would thrash. On the self-hosted
2×3090 runner, prefer a persistent `SCCACHE_DIR` so each new-release rebuild is incremental.

## Tag scheme

- GH Release: `v<vllm>-ampere` (e.g. `v0.23.0-ampere`); wheel keeps the real vLLM version + abi3.
- ghcr image: `ghcr.io/<owner>/vllm-ampere-optimized:<vllm>-ampere-<cu>` (+ moving `:latest`),
  e.g. `:v0.23.0-ampere-cu130`. cu129 variant: `:v0.23.0-ampere-cu129`.
