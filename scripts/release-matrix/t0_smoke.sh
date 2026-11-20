#!/bin/bash
# T0 -- smoke: what the image claims to be, the FA2 capture contract, default-env serve, the runner
# pin check, sampling-param limits and the Humming W4A8 fallback fix.
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
import humming; print("humming", getattr(humming, "__version__", "?"))
'

say "T0.1b package completeness: every vendored vllm/*.py is in the image"
# vllm/.gitignore matches some real modules (*_hip*); one missing from the vendored tree failed every
# serve of the first 0.34 build at model inspection.
oneshot t0-files -v "$(pwd)/../../vllm/vllm":/src:ro -- -c '
import pathlib, vllm
src, dst = pathlib.Path("/src"), pathlib.Path(vllm.__file__).parent
missing = sorted(str(p.relative_to(src)) for p in src.rglob("*.py") if not (dst / p.relative_to(src)).exists())
print(f"  missing from image: {len(missing)} (expect 0)", missing[:8])
'

say "T0.2 FA2 paged fwd_kvcache (the fork's spec-verify path): graph capture, no eager host sync, bounds guard"
oneshot t0-fa2cap -v "$(pwd)/fa2_capture_probe.py":/probe.py:ro -- /probe.py

say "T0.3 default env -- Qwen3-8B dense"
if serve t0-def 8180 Qwen3-8B; then
  backends t0-def; canary 8180
  # Over-limit sampling params once raised inside the V2 worker and killed the engine (upstream PR
  # #55164, ported): each must come back as a 400 and the next normal request must still succeed.
  python3 - <<'PY' >> "$LOG" 2>&1
import json, urllib.error, urllib.request
def post(extra):
    b = json.dumps({"model": "test", "prompt": "Hello", "max_tokens": 4, **extra}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8180/v1/completions", b,
                                   {"Content-Type": "application/json"}), timeout=120)
        return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return type(e).__name__
cases = {"logit_bias x2000": {"logit_bias": {str(i): 1.0 for i in range(2000)}},
         "allowed_token_ids x2000": {"allowed_token_ids": list(range(2000))},
         "stop_token_ids x200 + min_tokens": {"stop_token_ids": list(range(200)), "min_tokens": 1}}
for name, extra in cases.items():
    print(f"  over-limit {name}: HTTP {post(extra)} (expect 400)")
print(f"  normal request after: HTTP {post({})} (expect 200)")
PY
fi
matrix_rm t0-def

say "T0.4 runner: MRv2 is the only runner since vLLM 0.34; the image must not pin VLLM_USE_V2_MODEL_RUNNER"
# Any value but 1 aborts startup on 0.34, so a stale ENV left in a Dockerfile kills every serve.
echo "  image env pin: $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$IMG" \
  | grep -c '^VLLM_USE_V2_MODEL_RUNNER=') (expect 0)" >> "$LOG"

say "T0.5 Humming W4A8 35B-A3B MoE loads (crashes at load without the carried #68194 fix)"
if serve t0-hum 8180 Qwen3.6-35B-A3B-W4A16 "${HUM_ENV[@]}" --ARGS-- "${HUM_ARGS[@]}"; then
  canary 8180
fi
matrix_rm t0-hum

echo -e "\nT0_DONE" >> "$LOG"
