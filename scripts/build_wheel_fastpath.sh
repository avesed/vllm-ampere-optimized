#!/usr/bin/env bash
# Build a pip-installable wheel from the patched ./vllm checkout WITHOUT compiling CUDA,
# by reusing upstream's prebuilt kernels (VLLM_USE_PRECOMPILED). Valid only because our
# patch series is pure-Python (enforced by the native-code guard in apply_patches.sh).
set -euo pipefail

if [ "${NATIVE_CHANGED:-0}" = "1" ]; then
  echo "::error::NATIVE_CHANGED=1 — a patch touched native code; use the from-source build, not the fast-path"
  exit 1
fi

cd vllm
python -m pip install -U pip
# build deps incl. the hard torch==2.11.0 pin (so --no-build-isolation can reuse them)
python -m pip install -r requirements/build/cuda.txt

mkdir -p ../dist
# VLLM_USE_PRECOMPILED downloads the matching prebuilt .so; our patched *.py get bundled.
VLLM_USE_PRECOMPILED=1 python -m pip wheel . --no-deps --no-build-isolation -w ../dist

echo "== built wheel(s) =="
ls -la ../dist
