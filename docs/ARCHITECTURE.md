# Architecture

The maintainer's mental model. Why this repo is shaped the way it is.

## Overlay, not a fork

This repo vendors **zero** vLLM source. It is a *patch-overlay build recipe*: CI checks out
upstream `vllm-project/vllm` at a release tag, applies `patches/*.patch`, builds, publishes.

Why not a real GitHub fork carrying a patch branch:

- The standard fork-sync actions sync **branches, not tags** — and vLLM releases are tags
  (`v0.23.0`). A fork's only structural perk (`on: push: tags`) never fires without bolting on
  a tag-mirror step, at which point a `gh api .../releases/latest` compare is simpler and works
  identically here.
- Our entire change set is **3 pure-Python files**. Re-applying a tiny diff per release beats
  rebasing a long-lived branch against the whole upstream tree every two weeks.
- Pure-Python patches unlock a no-compile **fast-path wheel** (see below), awkward to reason
  about inside a fork's merge history.
- Precedent: conda-forge feedstocks, AUR PKGBUILDs, and other vLLM rebuilds all use the thin
  fetch→patch→build recipe model, not a vendored fork.

A true fork would only pay off if we ever carried many deep, intertwined **native** (`.cu`/CMake)
changes. The W4A8 Marlin patch is the opposite of that.

## Two build paths

| path | runner | speed | when |
|---|---|---|---|
| **fast-path wheel** (`VLLM_USE_PRECOMPILED=1`) | `ubuntu-latest` | minutes, no compile | always — the pip artifact |
| **overlay image** (`FROM vllm/vllm-openai` + patch) | `ubuntu-latest` | ~1–2 min, no compile | always — the shipped `:latest` image |
| **from-source single-arch image** (upstream `docker/Dockerfile`) | self-hosted GPU / larger runner | full compile | opt-in (`BUILD_RUNNER`); smaller artifact, or once a patch needs native code |

Both default paths reuse **official upstream builds** — the wheel pulls upstream's prebuilt `.so`,
the overlay starts from upstream's published image. We self-compile only on demand. **Why no
self-compile by default:** for a pure-Python patch it gives zero inference speedup — a fatbin loads
only the cubin matching the running SM, and upstream already ships sm_86 SASS; single-arch only
shrinks the artifact and build time. A **native-code guard** in `scripts/apply_patches.sh` flips the
from-source path on automatically if a patch ever edits `.cu`/`.cpp`/CMake (else the prebuilt
kernels would be stale — a silent correctness bug).

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
  Ampere hosts. Produce it by dispatching `build.yml` with `cuda_version=12.9.1`.
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
