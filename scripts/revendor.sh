#!/usr/bin/env bash
# Re-vendor the fork onto new upstream tags (the vendored-model replacement for apply_patches.sh).
# Clones fresh upstream into the vendored vllm/ + flashinfer/ trees and replays the edit recipe:
#   vLLM:       patches/regenerate.py (0001 W4A8, anchor-based) + patches/0002 (native marlin, git apply)
#               + drop in patches/flashinfer_int8/int8qk_backend.py
#   FlashInfer: patches/flashinfer_int8/apply_to_source.py (int8-QK, strict anchors)
# Any anchor/patch that no longer applies FAILS LOUDLY = the version-skew signal (refresh the recipe).
# After it runs, review `git diff`, bump nothing else, and open a PR to main — merging triggers the
# from-source build (.github/workflows/build.yml). Usage:
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
cp patches/flashinfer_int8/int8qk_backend.py \
   vllm/vllm/v1/attention/backends/int8qk_backend.py

echo "== re-vendor FlashInfer $FI_TAG =="
rm -rf flashinfer
git clone --depth 1 --branch "$FI_TAG" https://github.com/flashinfer-ai/flashinfer.git flashinfer
rm -rf flashinfer/.git
python3 patches/flashinfer_int8/apply_to_source.py                              # int8-QK (strict anchors)

echo "$VLLM_TAG" > UPSTREAM_VLLM_VERSION
echo
echo "re-vendored: vllm@$VLLM_TAG + flashinfer@$FI_TAG. Review 'git diff', then open a PR to main."
