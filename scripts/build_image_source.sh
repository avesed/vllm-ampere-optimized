#!/usr/bin/env bash
# Build the Ampere fork image FROM SOURCE from the VENDORED vllm/ tree — no upstream clone, no
# patch-apply step of any kind (the vendored vllm/ IS the fork, every edit is already in it): vLLM
# with the FA2 spec-verify routing baked in, built by upstream vllm/docker/Dockerfile.
# Push the final image to ghcr (:<tag>-ampere-<cu> + :latest).
#
# THIS IS THE RELEASE TOOL: run it YOURSELF on a local CUDA box and it pushes to ghcr. There is no CI
# auto-build — a self-hosted GPU runner on a public repo is a security risk. Needs docker buildx + a
# ghcr login (`docker login ghcr.io`) with packages:write. Optionally smoke-test after:
#   scripts/smoke_test.sh <pushed-image>  &&  scripts/ampere_kernel_ci.sh <pushed-image> "$(cat UPSTREAM_VLLM_VERSION)"
# Env: OWNER (required), CUDA_VERSION, TORCH_CUDA_ARCH_LIST, PUSH (1=push ghcr [default], 0=--load
# local for testing). VLLM_TAG defaults to UPSTREAM_VLLM_VERSION.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${OWNER:?set OWNER}"
CUDA_VERSION="${CUDA_VERSION:-13.0.2}"
# RELEASE = full multi-arch (matches upstream vLLM's Dockerfile default) so the published image runs
# on Turing..Blackwell, not just Ampere. For fast Ampere-only iteration use scripts/build_image_dev.sh.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX}"
VLLM_TAG="${VLLM_TAG:-$(cat UPSTREAM_VLLM_VERSION)}"   # the pinned vendored vLLM version
IMAGE="ghcr.io/${OWNER,,}/vllm-ampere-optimized"       # ghcr path must be lowercase
CU="cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
# parallel compile TUs (bounds build RAM). Default = min(nproc, 8) for safety on unknown hosts; set
# JOBS env to override (honored up to nproc) on a big build box with RAM headroom.
# The default cap of 8 is a floor for small boxes, not a target: on a 64-core host it turns the
# CUDA compile into a ~4.7h serial-ish crawl. Override with JOBS=<n> (and NVCC_THREADS, which only
# needs to cover the arch count -- 2 is enough for an Ampere-only list). Watch RAM: heavy template
# units run ~2GB per job.
if [ -n "${JOBS:-}" ]; then NPROC=$(nproc); [ "$JOBS" -gt "$NPROC" ] && JOBS=$NPROC
else JOBS=$(nproc); [ "$JOBS" -gt 8 ] && JOBS=8; fi
NVCC_THREADS="${NVCC_THREADS:-4}"
# GHA registry cache only works inside GitHub Actions; locally use docker's own layer cache.
GHA_CACHE=""; [ -n "${GITHUB_ACTIONS:-}" ] && GHA_CACHE="--cache-from type=gha --cache-to type=gha,mode=max"
# PUSH=1 (default) pushes the final image to ghcr; PUSH=0 keeps it in the LOCAL docker for testing
# without publishing. The build always --loads; the push is a plain docker push of the tagged image.
PUSH="${PUSH:-1}"
# Use the docker-driver "default" builder: it builds straight into the local image store, which the
# retag + push below rely on. CI (GHA cache) needs the container driver, so leave BUILDER empty there.
BUILDER="${BUILDER:---builder default}"; [ -n "${GITHUB_ACTIONS:-}" ] && BUILDER=""

[ -f vllm/docker/Dockerfile ] || { echo "::error::vendored vllm/ source missing (vllm/docker/Dockerfile)"; exit 1; }

# The upstream vLLM Dockerfile bind-mounts vllm/.git (setuptools-scm version + build commit), but the
# vendored vllm/ has no .git (it is a plain source tree). Synthesize an ephemeral one tagged VLLM_TAG so the
# build + the git-derived version resolve. Build-time only; never committed to the fork.
if [ ! -d vllm/.git ]; then
  echo "== synthesizing ephemeral vllm/.git tagged ${VLLM_TAG} (for the Dockerfile's git-version mount) =="
  git -C vllm init -q
  git -C vllm add -A
  git -C vllm -c user.email=build@local -c user.name=fork-build commit -qm "vendored ${VLLM_TAG}"
  git -C vllm tag -f "${VLLM_TAG}" >/dev/null
fi

VLLM_IMG="${IMAGE}:${VLLM_TAG}-vllm-${CU}"             # intermediate (vLLM-only) tag
# Tagging channel: CHANNEL=dev -> :dev only (do NOT move :latest). Default
# (release) -> :<VLLM_TAG>-ampere-<cu> + :latest (the existing release scheme).
CHANNEL="${CHANNEL:-}"
if [ -n "$CHANNEL" ]; then
  FINAL="${IMAGE}:${CHANNEL}"
  LATEST_TAG_ARG=""
else
  FINAL="${IMAGE}:${VLLM_TAG}-ampere-${CU}"
  LATEST_TAG_ARG="--tag ${IMAGE}:latest"
fi

echo "== build vLLM from vendored fork source ($TORCH_CUDA_ARCH_LIST) =="
docker buildx build vllm $BUILDER \
  --file vllm/docker/Dockerfile \
  --target vllm-openai \
  --platform linux/amd64 \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
  --build-arg max_jobs="$JOBS" \
  --build-arg nvcc_threads="$NVCC_THREADS" \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg VLLM_SCM_VERSION="${VLLM_TAG#v}" \
  --build-arg VLLM_BUILD_COMMIT="${GITHUB_SHA:-unknown}" \
  --build-arg VLLM_IMAGE_TAG="${VLLM_TAG}-ampere-${CU}" \
  --provenance=false \
  $GHA_CACHE \
  --tag "$VLLM_IMG" --load

# Final tags. (The famp_marlin stage and the vendored-FlashInfer overlay were retired with the vLLM
# 0.34 move: Humming serves W4A8, and FlashInfer is the stock wheel vLLM pins.)
docker tag "$VLLM_IMG" "$FINAL"
[ -n "$LATEST_TAG_ARG" ] && docker tag "$VLLM_IMG" "${IMAGE}:latest"
if [ "$PUSH" = 1 ]; then
  docker push "$FINAL"
  [ -n "$LATEST_TAG_ARG" ] && docker push "${IMAGE}:latest"
fi

echo "$([ "$PUSH" = 1 ] && echo pushed || echo 'built (local)') $FINAL${LATEST_TAG_ARG:+ + :latest}"
