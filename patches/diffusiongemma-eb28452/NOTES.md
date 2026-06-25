# DiffusionGemma int8 — patch rebase onto the #45163 base (staging)

Goal: run DiffusionGemma-26B-A4B (vLLM PR #45163) on the fork's INT8 Marlin path
(W4A16 weights + per-token int8 activations) on Ampere. #45163 merged to vLLM `main`
**after** the pinned `v0.23.0` tag, so the fork patches must be rebased onto a
#45163-containing base before re-vendoring + a from-source rebuild.

## Base
- vLLM commit `eb28452b10a1376d143b2847a78b31726db346dd` (the #45163 merge commit, merged 2026-06-12).
  Minimal churn past v0.23.0; `diffusion_gemma.py` present + in the model registry.
- No `v0.24.0` release tag exists yet — #45163 is `main`/nightly only. When a release tag ships,
  re-rebase against it for a clean ship (this `eb28452` snapshot is for validation).

## Trial-rebase result (verified all 6 apply clean in `revendor.sh` order on eb28452)
- **0001 / 0003 / 0004** — apply unchanged (no rebase needed).
- **0002 (8-row decode)** — REBASED: paths only. Upstream moved the marlin csrc into a stable-ABI
  subtree: `csrc/quantization/marlin/` → `csrc/libtorch_stable/quantization/marlin/`. Content unchanged.
- **0005 (int8-act MoE per-expert)** — REBASED:
  - csrc paths `csrc/moe/marlin_moe_wna16/` → `csrc/libtorch_stable/moe/marlin_moe_wna16/`.
  - `marlin_template.h` (CUDA kernel un-gate) + the 5 Python files: content unchanged, applied clean.
  - `ops.cu` (host binding) was rewritten upstream to the **stable ABI** (`STD_TORCH_CHECK`,
    `torch::stable::Tensor`, `mutable_data_ptr()`), so the 0005 ops.cu hunk was **re-ported** (same
    intent): widen the `global_scale` check `nvfp4 -> nvfp4 || a_type==kS8`; add the
    `global_scale_kernel_ptr = numel()>0 ? mutable_data_ptr() : nullptr` null-guard; swap the
    `marlin_mm(...)` global_scale arg. Logic identical to the v0.23.0 0005.
- **0006 (--marlin-input-dtype CLI)** — REBASED: re-anchored the 3 insertions in `arg_utils.py`
  around the new `diffusion_config` field that #45163 added (content of the inserts unchanged).

## ⚠️ NOT yet verified (needs the actual build)
Patch *applicability* is proven; **semantic/compile correctness is not** — only a from-source rebuild
confirms it. Two residual risks: (1) vendoring an eb28452 main snapshot pulls all v0.23.0→main churn
(envs.py / marlin_utils.py / compressed_tensors / caching.py also changed where our patches sit but
applied syntactically); (2) the `libtorch_stable` ABI migration — the ops.cu re-port compiles against
an API surface not yet test-compiled here.

## How to use (when ready to build — needs explicit go + HOST)
On `dev-dllm`, swap the three active patches for these REBASED ones, set the base, re-vendor, rebuild:
- `cp .../0002...REBASED.patch patches/0002-marlin-int8-8row-decode-ampere.patch` (and 0005, 0006)
- `echo eb28452b10a1376d143b2847a78b31726db346dd > UPSTREAM_VLLM_VERSION` (revendor.sh clones by ref)
- `scripts/revendor.sh <ref> <flashinfer_tag>` → review `git diff` → `OWNER=<you> scripts/build_image_source.sh` (HOST, ~2.5h)
- then `scripts/diffusiongemma_int8_validate.sh <image> <w4a16-ckpt>` (needs an LLM-Compressor W4A16 DiffusionGemma ckpt; 704-dim experts → fake-quant pre-check first).

Reproduce the trial in a scratch clone: `git fetch --depth 1 origin <ref>; git checkout FETCH_HEAD`,
then apply the patches (these REBASED ones for 0002/0005/0006).
