#!/usr/bin/env bash
# Tier-A anti-regression: run upstream mamba/GatedDeltaNet/causal-conv1d Triton KERNEL tests on the
# Ampere GPU runner (sm_80/sm_86) against the SHIPPED image. These kernels are vendored JIT-Triton in
# vLLM (vllm/model_executor/layers/{fla,mamba}/ops) — `git apply --check` and even a source build prove
# nothing about whether they compile or are numerically correct on Ampere/this-torch/this-triton/cu13.
# This catches Ampere codegen/numeric regressions after each upstream/torch/triton bump. Validated on a
# real RTX 3090 (sm_86): 228 + 286 + ... cases pass.
#
# Skips gracefully without a GPU — run it locally on an Ampere GPU box after build_image_source.sh.
# Usage: ampere_kernel_ci.sh <image> <vllm_tag>
set -euo pipefail

IMG="${1:?usage: ampere_kernel_ci.sh <image> <vllm_tag>}"
TAG="${2:?usage: ampere_kernel_ci.sh <image> <vllm_tag>}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::no GPU here — skipping Ampere kernel CI for $IMG"
  echo "(run this on an Ampere GPU box after building the image locally)"
  exit 0
fi

# Vetted on sm_86 — all pass (test_gdn_forward_core_split is sm_86-gated/skipped, so excluded).
TESTS=(
  tests/kernels/test_fused_gdn_post_conv.py
  tests/kernels/mamba/test_causal_conv1d.py
  tests/kernels/mamba/test_mamba_ssm.py
  tests/kernels/mamba/test_mamba_ssm_ssd.py
  tests/kernels/mamba/test_ssu_dispatch.py
  tests/kernels/mamba/test_mamba_ssm_configs.py
)

# Fetch ONLY tests/ for the matching tag on the host (the shipped image doesn't carry tests/).
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone --depth 1 --branch "$TAG" --filter=blob:none --sparse \
  https://github.com/vllm-project/vllm.git "$work/vllm"
git -C "$work/vllm" sparse-checkout set tests

docker pull "$IMG"
# NOTE: the official vllm image exposes `python3`, not `python` (no `python` alias).
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash \
  -v "$work/vllm/tests:/ampere-ci/tests:ro" "$IMG" -lc '
    set -e
    python3 -m pip install -q pytest einops tblib
    cp -r /ampere-ci/tests /tmp/tests   # writable copy (pytest cache / sparse-checkout is ro)
    cd /tmp
    python3 -m pytest -q --no-header -p no:cacheprovider \
      '"${TESTS[*]}"'
  '

echo "Ampere kernel CI passed for $IMG ($TAG)"
