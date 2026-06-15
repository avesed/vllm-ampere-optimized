#!/usr/bin/env bash
# Build a MoE-CAPABLE wheel by OVERLAYING our pure-Python patch series onto the OFFICIAL
# upstream vllm wheel. The official wheel ships vllm/_moe_C.abi3.so (+ every kernel) and runs
# on cu130 / torch 2.11; VLLM_USE_PRECOMPILED instead fetches a stable-ABI .so subset that
# DROPS _moe_C and silently BREAKS every MoE model (torch.ops._moe_C.topk_softmax missing).
# This mirrors docker/Dockerfile.overlay (FROM vllm/vllm-openai). Zero CUDA compile, any runner.
# Usage: build_wheel_overlay.sh <vllm version, e.g. 0.23.0>   (strip the leading 'v')
set -euo pipefail

VER="${1:?usage: build_wheel_overlay.sh <vllm version, e.g. 0.23.0>}"
VER="${VER#v}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
mkdir -p "$REPO_ROOT/dist"

python -m pip install -q wheel

echo "== download official vllm==$VER wheel (ships _moe_C) =="
python -m pip download "vllm==$VER" --no-deps -d "$WORK/dl"
WHL="$(ls "$WORK"/dl/vllm-*.whl)"
python - "$WHL" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
assert any("vllm/_moe_C" in n for n in names), "official wheel lacks _moe_C — overlay would not fix MoE!"
print("  official wheel contains _moe_C.abi3.so  ✓")
PY

echo "== unpack =="
( cd "$WORK" && python -m wheel unpack "$WHL" )
UNP="$(ls -d "$WORK"/vllm-*/ | head -1)"

echo "== overlay our pure-Python patches (apply only vllm/ hunks; requirements/* etc. are N/A in a wheel) =="
( cd "$UNP"
  for p in "$REPO_ROOT"/patches/*.patch; do
    if git apply -p1 --include='vllm/*' --whitespace=fix "$p" 2>/dev/null; then
      echo "  applied $(basename "$p")"
    else
      echo "  (no applicable vllm/ hunks in $(basename "$p"))"
    fi
  done
  if compgen -G "$REPO_ROOT/configs/fused_moe/*.json" >/dev/null; then
    cp "$REPO_ROOT"/configs/fused_moe/*.json vllm/model_executor/layers/fused_moe/configs/
    echo "  dropped in device-tuned fused-MoE configs"
  fi
)

echo "== repack (wheel pack regenerates RECORD with the patched hashes) =="
python -m wheel pack "$UNP" -d "$REPO_ROOT/dist"

echo "== built MoE-capable overlay wheel =="
ls -la "$REPO_ROOT/dist"
echo "NOTE: the official wheel already pins fastapi<0.137 (that cap landed upstream in v0.23.0,"
echo "      so the former patch 0002 is no longer carried) — no extra fastapi handling needed."
