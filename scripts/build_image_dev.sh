#!/usr/bin/env bash
# FAST DEV build — for iteration, NOT release. Two levers vs scripts/build_image_source.sh:
#   1. sm_86 ONLY (TORCH_CUDA_ARCH_LIST=8.6) — the dev box is 3090/sm_86; release does "8.0 8.6".
#   2. FA3 (Hopper sm_90) DISABLED — FA3 is ~59% of the CUDA compile (199/338 objects) and is DEAD
#      WEIGHT on Ampere (runtime uses FA2). The FA repo hardcodes `set(FA3_ENABLED ON)` unconditionally
#      and the vLLM Dockerfile exposes no cmake/VLLM_FLASH_ATTN_SRC_DIR passthrough, so we inject a
#      FetchContent PATCH_COMMAND into the vendored vllm_flash_attn.cmake that seds FA3_ENABLED -> OFF.
#      (Reverted after the build; release builds keep FA3 ON for completeness.)
# Net: ~half-or-less the build time of a full release build. PUSH=0 by default (local --load image).
#
# ⚠️ FIRST-USE VALIDATION: the FA3 PATCH_COMMAND injection is new — the first dev build is itself the
#    test that (a) cmake accepts the PATCH_COMMAND and (b) the image still serves (FA2 path intact).
#
# Usage:  OWNER=<you> scripts/build_image_dev.sh        # local sm86 image, FA3 off, no push
# Env:    OWNER (required), CUDA_VERSION, TORCH_CUDA_ARCH_LIST (default 8.6), PUSH (default 0),
#         VLLM_TAG (default "<UPSTREAM>-dev"), JOBS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${OWNER:?set OWNER}"

FA_CMAKE="vllm/cmake/external_projects/vllm_flash_attn.cmake"
[ -f "$FA_CMAKE" ] || { echo "::error::vendored $FA_CMAKE missing (run revendor first)"; exit 1; }

# --- disable FA3 for this build only (backup + restore on exit) ---
cp "$FA_CMAKE" "$FA_CMAKE.devbak"
trap 'mv -f "$FA_CMAKE.devbak" "$FA_CMAKE" 2>/dev/null || true' EXIT
python3 - "$FA_CMAKE" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "GIT_PROGRESS TRUE\n"
inject = ('GIT_PROGRESS TRUE\n'
          '          # DEV build: skip FA3 (Hopper sm90) — dead weight on Ampere, ~59% of the compile.\n'
          '          PATCH_COMMAND sed -i "s/set(FA3_ENABLED ON)/set(FA3_ENABLED OFF)/" CMakeLists.txt\n')
if "FA3_ENABLED OFF" in s:
    print("FA3 already disabled in", p)
elif s.count(anchor) == 1:
    open(p, "w").write(s.replace(anchor, inject))
    print("injected FA3-disable PATCH_COMMAND into", p)
else:
    sys.exit(f"anchor 'GIT_PROGRESS TRUE' found {s.count(anchor)}x (expected 1) — FA cmake drifted; "
             "update build_image_dev.sh")
PY

# --- delegate the actual build to the release script with dev env ---
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PUSH="${PUSH:-0}"
export VLLM_TAG="${VLLM_TAG:-$(cat UPSTREAM_VLLM_VERSION)-dev}"
echo "== DEV build: arch=$TORCH_CUDA_ARCH_LIST  FA3=off  tag=$VLLM_TAG  push=$PUSH =="
bash scripts/build_image_source.sh
echo "== DEV build done (FA3-disable reverted on exit). For a full release image use build_image_source.sh. =="
