#!/usr/bin/env bash
# Build the Ampere (sm_80 + sm_86) runtime image from the PATCHED ./vllm checkout using
# upstream docker/Dockerfile (unmodified, driven entirely by --build-arg) and push to ghcr.
# The patches were already applied to the working tree by apply_patches.sh, so they bake
# into the wheel during the build. Env: OWNER, CU, VLLM_TAG, CUDA_VERSION, TORCH_CUDA_ARCH_LIST.
set -euo pipefail

: "${OWNER:?}"; : "${CU:?}"; : "${VLLM_TAG:?}"; : "${CUDA_VERSION:?}"; : "${TORCH_CUDA_ARCH_LIST:?}"
IMAGE="ghcr.io/${OWNER,,}/vllm-ampere-optimized"   # ghcr requires a lowercase path

JOBS="$(nproc)"; [ "$JOBS" -gt 8 ] && JOBS=8        # cap parallel TUs to bound RAM

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
  --cache-from type=gha \
  --cache-to type=gha,mode=max \
  --tag "${IMAGE}:${VLLM_TAG}-ampere-${CU}" \
  --tag "${IMAGE}:latest" \
  --push

echo "pushed ${IMAGE}:${VLLM_TAG}-ampere-${CU} (+ :latest)"
