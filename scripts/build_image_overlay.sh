#!/usr/bin/env bash
# Default image build: official upstream vllm/vllm-openai:<tag> + our pure-Python patch + configs,
# ZERO CUDA compile (~1-2 min, any runner). Tags :<tag> and :latest. Env: OWNER, VLLM_TAG.
set -euo pipefail

: "${OWNER:?}"; : "${VLLM_TAG:?}"
IMAGE="ghcr.io/${OWNER,,}/vllm-ampere-optimized"     # ghcr requires a lowercase path
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker buildx build "$REPO_ROOT" \
  --file "$REPO_ROOT/docker/Dockerfile.overlay" \
  --build-arg BASE="vllm/vllm-openai:${VLLM_TAG}" \
  --provenance=false \
  --tag "${IMAGE}:${VLLM_TAG}" \
  --tag "${IMAGE}:latest" \
  --push

echo "pushed ${IMAGE}:${VLLM_TAG} (+ :latest)  [overlay, zero-compile]"
