#!/bin/bash
# T0 -- smoke: what the image claims to be, the FA2 capture contract, default-env serve, the V2
# escape hatch, and the in-image famp dispatch unit tests.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t0.log; : > "$LOG"; . ./lib.sh

oneshot(){ # name [docker args...] -- python args...
  local name=$1; shift
  local dargs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do dargs+=("$1"); shift; done; shift
  matrix_rm "$name"
  docker run --rm --name "$name" --label "$MATRIX_LABEL" --gpus device=0 "${dargs[@]}" \
    --entrypoint python3 "$IMG" "$@" 2>&1 | grep -av -i "warn" | tail -12 >> "$LOG"
}

say "T0.1 versions + plugin entry points"
oneshot t0-ver -- -c '
import importlib.metadata as md, torch, flashinfer, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| flashinfer", flashinfer.__version__)
eps = sorted(e.name for e in md.entry_points(group="vllm.general_plugins"))
print("general_plugins:", eps)
import flashampere.marlin.kernel as k; print("fampmarlin kernel import OK:", k.FampMarlinKernel.__name__)
import vllm.v1.attention.backends.flashampere.kernels as fk; print("famp kernels import OK")
'

say "T0.2 FA2 paged fwd_kvcache (the fork's spec-verify path): graph capture, no eager host sync, bounds guard"
oneshot t0-fa2cap -v "$(pwd)/fa2_capture_probe.py":/probe.py:ro -- /probe.py

say "T0.3 default env (famp off) -- Qwen3-8B dense"
if serve t0-def 8180 Qwen3-8B; then
  backends t0-def; fired t0-def; canary 8180
fi
matrix_rm t0-def

say "T0.4 V2 escape hatch (VLLM_USE_V2_MODEL_RUNNER=1)"
if serve t0-v2 8180 Qwen3-8B -e VLLM_USE_V2_MODEL_RUNNER=1; then
  docker logs t0-v2 2>&1 | grep -aiE "model runner v2|GPUModelRunnerV2|gpu/model_runner" | head -2 >> "$LOG"
  canary 8180
fi
matrix_rm t0-v2

say "T0.7 famp dispatch unit tests against the in-image flashampere"
# The tests import the repo-root `flashampere` package; in the image famp lives inside vLLM and
# /opt/famp/flashampere holds only the marlin plugin, so alias the in-image package before collection.
oneshot t0-pytest -v "$(pwd)/../../vllm/tests/v1/attention":/t:ro -- -c '
import importlib, pkgutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=False)
src = "vllm.v1.attention.backends.flashampere"
pkg = importlib.import_module(src)
sys.modules["flashampere"] = pkg
for m in pkgutil.iter_modules(pkg.__path__):
    try:
        sys.modules["flashampere." + m.name] = importlib.import_module(src + "." + m.name)
    except Exception as e:
        print("alias skipped:", m.name, type(e).__name__, e)
import pytest
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "/t/test_flashampere_dispatch.py",
                      "/t/test_flashampere_kv_layout.py", "/t/test_flashampere_hd512_gate.py"]))
'
echo -e "\nT0_DONE" >> "$LOG"
