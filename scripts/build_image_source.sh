#!/usr/bin/env bash
# Build the COMPLETE Ampere int8 fork image FROM SOURCE from the VENDORED trees — no upstream
# clone, no apply_patches (the vendored vllm/ + flashinfer/ ARE the fork, all edits baked in):
#   stage 1: vLLM (W4A8 Marlin + int8-8row decode + int8qk_backend, baked into vllm/) from source
#            via upstream vllm/docker/Dockerfile -> sm_80 + sm_86 fatbin.
#   stage 2: overlay the vendored int8-QK flashinfer/ (docker/Dockerfile.flashinfer-int8).
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
if [ -n "${JOBS:-}" ]; then NPROC=$(nproc); [ "$JOBS" -gt "$NPROC" ] && JOBS=$NPROC
else JOBS=$(nproc); [ "$JOBS" -gt 8 ] && JOBS=8; fi
# GHA registry cache only works inside GitHub Actions; locally use docker's own layer cache.
GHA_CACHE=""; [ -n "${GITHUB_ACTIONS:-}" ] && GHA_CACHE="--cache-from type=gha --cache-to type=gha,mode=max"
# PUSH=1 (default) pushes the final image to ghcr; PUSH=0 builds it into the LOCAL docker (--load) for
# testing without publishing. Stage-1 is always --load (intermediate base for stage-2).
PUSH="${PUSH:-1}"; [ "$PUSH" = 1 ] && PUSH_FLAG="--push" || PUSH_FLAG="--load"
# Use the docker-driver "default" builder: the 2-stage build does stage-1 `--load` (into the local
# docker image store) then stage-2 `FROM` it — but a docker-CONTAINER-driver builder resolves FROM
# from the REGISTRY, so it can't see the local intermediate and stage-2 fails ("...vllm-cu130: not
# found"). The docker driver reads the local store + supports --load/--push for single-platform.
# CI (GHA cache) needs the container driver, so leave BUILDER empty there.
BUILDER="${BUILDER:---builder default}"; [ -n "${GITHUB_ACTIONS:-}" ] && BUILDER=""

[ -f vllm/docker/Dockerfile ] || { echo "::error::vendored vllm/ source missing (vllm/docker/Dockerfile)"; exit 1; }
[ -f flashinfer/include/flashinfer/mma.cuh ] || { echo "::error::vendored flashinfer/ source missing"; exit 1; }

# The upstream vLLM Dockerfile bind-mounts vllm/.git (setuptools-scm version + build commit), but the
# vendored vllm/ has no .git (revendor strips it). Synthesize an ephemeral one tagged VLLM_TAG so the
# build + the git-derived version resolve. Build-time only; never committed to the fork.
if [ ! -d vllm/.git ]; then
  echo "== synthesizing ephemeral vllm/.git tagged ${VLLM_TAG} (for the Dockerfile's git-version mount) =="
  git -C vllm init -q
  git -C vllm add -A
  git -C vllm -c user.email=build@local -c user.name=fork-build commit -qm "vendored ${VLLM_TAG}"
  git -C vllm tag -f "${VLLM_TAG}" >/dev/null
fi

VLLM_IMG="${IMAGE}:${VLLM_TAG}-vllm-${CU}"             # intermediate (vLLM-only) tag
FI_IMG="${IMAGE}:${VLLM_TAG}-fi-${CU}"                 # intermediate (vLLM + flashinfer) tag
# famp_marlin release arches (comma list; generate_kernels.py + build.py syntax). Default = Ampere
# sm_80+sm_86 (the fork's scope); FampMarlin gates selection to these, stock _C serves other arches.
FAMP_MARLIN_ARCH="${FAMP_MARLIN_ARCH:-8.0,8.6}"
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

echo "== stage 1/2: build vLLM from vendored fork source (sm_80+sm_86) =="
docker buildx build vllm $BUILDER \
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
  $GHA_CACHE \
  --tag "$VLLM_IMG" --load

echo "== stage 2/3: overlay vendored int8-QK flashinfer (load locally; the final push is stage 3) =="
docker buildx build . $BUILDER \
  --file docker/Dockerfile.flashinfer-int8 \
  --build-arg BASE="$VLLM_IMG" \
  --platform linux/amd64 \
  --provenance=false \
  --tag "$FI_IMG" --load

# stage 3: compile the vendored famp_marlin FROM SOURCE on the from-source image (NOT an overlay on an
# upstream wheel) + register the FampMarlinKernel plugin. P4 (the int8-act config widening) ships in the
# vendored vllm/ via patches/0007.
echo "== stage 3/3: compile vendored famp_marlin (sm: $FAMP_MARLIN_ARCH) + register plugin ($([ "$PUSH" = 1 ] && echo 'push to ghcr' || echo 'load locally')) =="
docker buildx build . $BUILDER \
  --file docker/Dockerfile.famp-marlin \
  --build-arg BASE="$FI_IMG" \
  --build-arg FAMP_MARLIN_ARCH="$FAMP_MARLIN_ARCH" \
  --platform linux/amd64 \
  --provenance=false \
  --tag "$FINAL" \
  $LATEST_TAG_ARG \
  $PUSH_FLAG

echo "$([ "$PUSH" = 1 ] && echo pushed || echo 'built (local)') $FINAL${LATEST_TAG_ARG:+ + :latest}  [complete from-source fork: W4A8 + int8-8row + int8-QK flashinfer + vendored famp_marlin]"
