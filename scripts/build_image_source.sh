#!/usr/bin/env bash
# Build the COMPLETE Ampere int8 fork image FROM SOURCE from the VENDORED trees — no upstream
# clone, no apply_patches (the vendored vllm/ + flashinfer/ ARE the fork, all edits baked in):
#   stage 1: vLLM (W4A8 Marlin + int8-8row decode + int8qk_backend, baked into vllm/) from source
#            via upstream vllm/docker/Dockerfile -> sm_80 + sm_86 fatbin.
#   stage 2: overlay the vendored int8-QK flashinfer/ (docker/Dockerfile.flashinfer-int8).
# Push the final image to ghcr (:<tag>-ampere-<cu> + :latest).
# Env: OWNER (required), CUDA_VERSION, TORCH_CUDA_ARCH_LIST. VLLM_TAG defaults to UPSTREAM_VLLM_VERSION.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${OWNER:?set OWNER}"
CUDA_VERSION="${CUDA_VERSION:-13.0.2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6}"
VLLM_TAG="${VLLM_TAG:-$(cat UPSTREAM_VLLM_VERSION)}"   # the pinned vendored vLLM version
IMAGE="ghcr.io/${OWNER,,}/vllm-ampere-optimized"       # ghcr path must be lowercase
CU="cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
JOBS="$(nproc)"; [ "$JOBS" -gt 8 ] && JOBS=8           # cap parallel TUs to bound build RAM

[ -f vllm/docker/Dockerfile ] || { echo "::error::vendored vllm/ source missing (vllm/docker/Dockerfile)"; exit 1; }
[ -f flashinfer/include/flashinfer/mma.cuh ] || { echo "::error::vendored flashinfer/ source missing"; exit 1; }

VLLM_IMG="${IMAGE}:${VLLM_TAG}-vllm-${CU}"             # intermediate (vLLM-only) tag
FINAL="${IMAGE}:${VLLM_TAG}-ampere-${CU}"

echo "== stage 1/2: build vLLM from vendored fork source (sm_80+sm_86) =="
docker buildx build vllm \
  --file vllm/docker/Dockerfile \
  --target vllm-openai \
  --platform linux/amd64 \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
  --build-arg max_jobs="$JOBS" \
  --build-arg nvcc_threads=4 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg VLLM_BUILD_COMMIT="${GITHUB_SHA:-unknown}" \
  --build-arg VLLM_IMAGE_TAG="${VLLM_TAG}-ampere-${CU}" \
  --provenance=false \
  --cache-from type=gha --cache-to type=gha,mode=max \
  --tag "$VLLM_IMG" --load

echo "== stage 2/2: overlay vendored int8-QK flashinfer + push final =="
docker buildx build . \
  --file docker/Dockerfile.flashinfer-int8 \
  --build-arg BASE="$VLLM_IMG" \
  --platform linux/amd64 \
  --provenance=false \
  --tag "$FINAL" \
  --tag "${IMAGE}:latest" \
  --push

echo "pushed $FINAL + :latest  [complete from-source fork: W4A8 + int8-8row + int8-QK flashinfer]"
