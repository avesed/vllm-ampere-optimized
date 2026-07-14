#!/usr/bin/env bash
# Re-vendor the fork onto new upstream tags (the vendored-model replacement for apply_patches.sh).
# Clones fresh upstream into the vendored vllm/ + flashinfer/ trees and replays the edit recipe:
#   vLLM:       patches/regenerate.py (0001 W4A8, anchor-based) + patches/0002 (native marlin, git apply)
#               + patches/0003 (AOT compile cache-key, git apply)
#               (the old patch-0004 int8qk standalone was REMOVED: int8-QK is net-negative)
#   FlashInfer: patches/flashinfer_int8/apply_to_source.py (int8-QK kernel — currently UNUSED dead code
#               after the int8-QK removal; kept until a separate flinfer revert lands)
# Any anchor/patch that no longer applies FAILS LOUDLY = the version-skew signal (refresh the recipe).
# There is NO CI auto-build (a self-hosted GPU runner on a public repo is a security risk). After it
# runs, review `git diff`, commit, then build + push the image YOURSELF locally:
#   OWNER=<you> scripts/build_image_source.sh
# Usage:
#   scripts/revendor.sh <vllm_tag> <flashinfer_tag>     e.g. scripts/revendor.sh v0.23.0 v0.6.12
set -euo pipefail

VLLM_TAG="${1:?usage: revendor.sh <vllm_tag> <flashinfer_tag>}"
FI_TAG="${2:?usage: revendor.sh <vllm_tag> <flashinfer_tag>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== re-vendor vLLM $VLLM_TAG =="
rm -rf vllm
git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm.git vllm
rm -rf vllm/.git
python3 patches/regenerate.py vllm                                              # 0001 (fails on drift)
git apply -p1 --directory=vllm patches/0002-marlin-int8-8row-decode-ampere.patch # 0002 native
git apply -p1 --directory=vllm patches/0003-aot-compile-cache-quant-scheme-key.patch # 0003 AOT cache-key
git apply -p1 --directory=vllm patches/0005-int8act-moe-perexpert-ampere.patch # 0005 int8-act MoE per-expert scale (kernel un-gate + python)
git apply -p1 --directory=vllm patches/0006-marlin-input-dtype-cli-alias.patch # 0006 --marlin-input-dtype CLI alias for VLLM_MARLIN_INPUT_DTYPE
git apply -p1 --directory=vllm patches/0007-famp-marlin-config.patch # 0007 widen the int8-act override to FampMarlinKernel (vendored famp_marlin, built FROM SOURCE in build_image_source.sh stage 3)

echo "== re-vendor FlashInfer $FI_TAG =="
rm -rf flashinfer
git clone --depth 1 --branch "$FI_TAG" https://github.com/flashinfer-ai/flashinfer.git flashinfer
rm -rf flashinfer/.git
python3 patches/flashinfer_int8/apply_to_source.py                              # int8-QK (strict anchors)

echo "$VLLM_TAG" > UPSTREAM_VLLM_VERSION
echo
echo "re-vendored: vllm@$VLLM_TAG + flashinfer@$FI_TAG. Review 'git diff', commit, then build + push"
echo "the image yourself locally:  OWNER=<you> scripts/build_image_source.sh   (no CI auto-build)."
