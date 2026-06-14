#!/usr/bin/env bash
# Clone upstream vLLM at a tag, apply our patch series (3-way), and flag whether any
# patch touched native code (which would make the precompiled fast-path unsafe).
# Leaves a patched ./vllm checkout. Usage: apply_patches.sh <vllm_tag>
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
# conflict with upstream), pure data, so they don't trip the native-code guard below
# and the precompiled fast-path stays valid.
shopt -s nullglob
moe_cfgs=("$REPO_ROOT"/configs/fused_moe/*.json)
if [ ${#moe_cfgs[@]} -gt 0 ]; then
  cp -v "${moe_cfgs[@]}" vllm/model_executor/layers/fused_moe/configs/
fi

echo "== native-code guard =="
# The fast-path (VLLM_USE_PRECOMPILED) ships upstream's prebuilt .so. That is correct
# ONLY while our patches touch zero native code. If a patch ever edits .cu/.cpp/CMake,
# the prebuilt kernels would be STALE -> force a from-source build instead.
if git diff --name-only HEAD | grep -E '\.(cu|cpp|cc|cuh|h)$|CMakeLists|(^|/)csrc/'; then
  echo "::warning::a patch touched native code -> fast-path unsafe; build from source"
  echo "NATIVE_CHANGED=1" >> "${GITHUB_ENV:-/dev/null}"
else
  echo "patches are pure-Python; precompiled fast-path is valid"
fi
