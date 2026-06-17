#!/usr/bin/env bash
# Clone upstream vLLM at a tag, apply our patch series (3-way), and flag whether any
# patch touched native code (which the default overlay wheel/image cannot carry — those kernels
# ship only via THIS from-source build). Leaves a patched ./vllm checkout. Usage: apply_patches.sh <vllm_tag>
set -euo pipefail

TAG="${1:?usage: apply_patches.sh <vllm_tag>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rm -rf vllm
git clone --depth 1 --branch "$TAG" https://github.com/vllm-project/vllm.git vllm
cd vllm

echo "== drift canary: git apply --check =="
if ! git apply --check "$REPO_ROOT"/patches/*.patch; then
  echo "::error::patch series does not apply to $TAG — refresh patches/ (docs/PATCHING.md)"
  exit 1
fi

echo "== applying patch series (3-way) =="
git apply --3way --whitespace=fix "$REPO_ROOT"/patches/*.patch

echo "== overlay device-tuned configs (new data files, not diffs) =="
# Drop our device-tuned fused-MoE Triton configs into the package. New files (never
# conflict with upstream), pure data, so they don't trip the native-code guard below.
shopt -s nullglob
moe_cfgs=("$REPO_ROOT"/configs/fused_moe/*.json)
if [ ${#moe_cfgs[@]} -gt 0 ]; then
  cp -v "${moe_cfgs[@]}" vllm/model_executor/layers/fused_moe/configs/
fi

echo "== native-code guard =="
# The default OVERLAY ship (wheel + image) carries ONLY vllm/* (Python) hunks, so native changes
# never reach it. If a patch edits .cu/.cpp/CMake, those kernels exist ONLY in THIS from-source
# build -> flag it so the release is known to need the from-source image for those kernels.
if git diff --name-only HEAD | grep -E '\.(cu|cpp|cc|cuh|h)$|CMakeLists|(^|/)csrc/'; then
  echo "::warning::a patch touched native code -> NOT in the overlay wheel/image; ships only via this from-source build"
  echo "NATIVE_CHANGED=1" >> "${GITHUB_ENV:-/dev/null}"
else
  echo "patches are pure-Python; the overlay wheel/image carries them fully"
fi
