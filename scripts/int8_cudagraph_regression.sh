#!/usr/bin/env bash
# Regression for patch 0003 (AOT compile cache-key). Two checkpoints of the SAME architecture but
# DIFFERENT quant schemes — pack-quantized **W4A16** + int-quantized **W4A8** — must NOT collide on
# the torch.compile AOT-compile on-disk cache. We run W4A16 then W4A8 under cudagraph against a
# SHARED compile cache (the exact collision condition): without 0003 the W4A8 run loads the W4A16's
# cached graph and dies `KeyError: 'weight_zero_point'`; with 0003 the cache key includes the quant
# scheme so the two stay distinct. This asserts the fork image actually carries a working 0003.
#
# Run locally on a GPU box after build_image_source.sh (there is no CI GPU). Skips gracefully w/o a
# GPU or w/o the checkpoints. Provide a same-architecture W4A16 + W4A8 pair (e.g. the validated
# Qwen3.5-9B-{w4a16,w4a8}-g32-awqmse):
#   W4A16_CKPT=/path/to/w4a16 W4A8_CKPT=/path/to/w4a8 int8_cudagraph_regression.sh <image>
# NOTE: the ckpt paths are passed verbatim to `docker run -v`, so they must be paths the DOCKER
# DAEMON sees. On a normal local box that's just the local path. Under docker-in-docker / sibling
# containers (the daemon is on the host), use the HOST path (an in-container path mounts EMPTY and
# the W4A8 run fails with a misleading exit 1 that looks like a 0003 regression).
set -euo pipefail

IMG="${1:?usage: int8_cudagraph_regression.sh <image>}"
# GPUS is passed verbatim to `docker run --gpus`. Use a valid spec: "all" or
# "device=0" / "device=0,1". NOTE: a bare index like GPUS=0 means "zero GPUs"
# to Docker (not GPU index 0) and yields a container with no CUDA device.
GPUS="${GPUS:-all}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::no GPU here — skipping int8 cudagraph regression for $IMG"
  echo "(run this on the GPU box after building the image locally)"
  exit 0
fi
if [ -z "${W4A16_CKPT:-}" ] || [ -z "${W4A8_CKPT:-}" ]; then
  echo "::warning::W4A16_CKPT / W4A8_CKPT unset — skipping int8 cudagraph regression"
  echo "(point both at a same-architecture W4A16 + W4A8 checkpoint pair to enable it)"
  exit 0
fi

CACHE="$(mktemp -d)"                     # ONE shared compile cache for both runs = the collision condition
trap 'rm -rf "$CACHE"' EXIT
LOG="$(mktemp)"

# Load under cudagraph (enforce_eager=False) so the AOT-compile cache is exercised; tiny gen.
PROBE='
from vllm import LLM, SamplingParams
llm = LLM(model="/m", enforce_eager=False, max_model_len=2048, gpu_memory_utilization=0.55,
          max_num_seqs=4, trust_remote_code=True, limit_mm_per_prompt={"image": 0, "video": 0})
o = llm.generate(["2+2="], SamplingParams(max_tokens=8, temperature=0.0))
print("PROBE_OK", repr(o[0].outputs[0].text[:40]))
'
run() {  # $1 = checkpoint host path
  docker run --rm --gpus "$GPUS" -e HOME=/cache -e VLLM_ENABLE_V1_MULTIPROCESSING=0 --shm-size=8g \
    -v "$CACHE":/cache -v "$1":/m:ro --entrypoint python3 "$IMG" -c "$PROBE"
}

echo "== [1/2] W4A16 — populates the shared AOT-compile cache (cudagraph) =="
run "$W4A16_CKPT"

echo "== [2/2] W4A8 — same cache; must NOT bind the W4A16 graph (cudagraph) =="
# Stream the full run to $LOG, then assert on the file. Do NOT pipe `run` into
# `grep -q`: PROBE_OK is printed before the container's teardown lines, so a
# matching `grep -q` closes the pipe early and SIGPIPEs `tee`; under `pipefail`
# that turns a clean PASS into a spurious exit 141 -> FAIL. Asserting on the
# captured file decouples the success check from the pipeline's exit status.
# `|| true`: a real collision makes the container exit non-zero; without this the
# pipeline's failure would trip `set -e`/`pipefail` and abort before the verdict.
# The grep over $LOG below is the authoritative pass/fail check.
run "$W4A8_CKPT" 2>&1 | tee "$LOG" || true
if grep -q "PROBE_OK" "$LOG"; then
  echo "PASS: W4A8 ran clean under cudagraph against the W4A16-populated cache (0003 active)."
  exit 0
else
  echo "::error::FAIL: W4A8 did not complete — AOT cache-key collision (0003 missing or broken in $IMG)."
  grep -iE "weight_zero_point|KeyError" "$LOG" | head -3 || true
  exit 1
fi
