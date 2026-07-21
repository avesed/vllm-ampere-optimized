#!/usr/bin/env bash
# R04 (probabilistic draft) x MTP A/B on the SANCTIONED path: fresh `vllm serve`
# per arm (draft_sample_method is launch-time), bench via api_bench_client.py
# (OpenAI API streaming; accept_len from /metrics num_drafts/num_accepted delta).
#
# Run ONE arm per invocation (each boots a fresh server, benches, tears it down):
#   scripts/r04_mtp_ab.sh <arm>
#   arm = nospec | greedy | prob | synth:<len>     (e.g. synth:1.8)
#
# Env (with defaults):
#   MODEL=/home/coder/models/Qwen3.6-27B-W4A16   # W4A16 ckpt; W4A8=1 adds int8-act
#   W4A8=1            # export VLLM_MARLIN_INPUT_DTYPE=int8 (production W4A8 decode)
#   TP=2 MAXLEN=34000 MAXSEQS=16 MAXBT=2048 GMU=0.90 PORT=8000
#   K=2              # num_speculative_tokens
#   REP=90 MAXTOK=512 # bench prompt length / decode tokens (accept-len sample)
#   EAGER=0          # 1 -> --enforce-eager on BOTH arms (use only if capture dies)
#
# Suggested sequence (run top-to-bottom, paste each RESULT line back):
#   ./r04_mtp_ab.sh synth:1.8   # harness calibration: accept_len should read ~1.8
#   ./r04_mtp_ab.sh nospec      # decode floor (accept_len=1.0 by construction)
#   ./r04_mtp_ab.sh greedy      # Arm A: baseline MTP accept-len
#   ./r04_mtp_ab.sh prob        # Arm B: R04 probabilistic draft
set -euo pipefail

ARM="${1:?arm required: nospec|greedy|prob|synth:<len>}"
MODEL="${MODEL:-/home/coder/models/Qwen3.6-27B-W4A16}"
W4A8="${W4A8:-1}"; TP="${TP:-2}"; MAXLEN="${MAXLEN:-34000}"; MAXSEQS="${MAXSEQS:-16}"
MAXBT="${MAXBT:-2048}"; GMU="${GMU:-0.90}"; PORT="${PORT:-8000}"; K="${K:-2}"
REP="${REP:-90}"; MAXTOK="${MAXTOK:-512}"; EAGER="${EAGER:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-240}"   # bump for cold torch.compile (new K/graph ~170s+)
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- build --speculative-config per arm -------------------------------------
SPEC=""; LABEL="$ARM"
case "$ARM" in
  nospec) SPEC="" ;;
  greedy) SPEC='{"method":"mtp","num_speculative_tokens":'"$K"'}' ;;            # draft_sample_method defaults greedy
  prob)   SPEC='{"method":"mtp","num_speculative_tokens":'"$K"',"draft_sample_method":"probabilistic"}' ;;
  synth:*) LEN="${ARM#synth:}"
           SPEC='{"method":"mtp","num_speculative_tokens":'"$K"',"rejection_sample_method":"synthetic","synthetic_acceptance_length":'"$LEN"'}'
           LABEL="synth_$LEN" ;;
  *) echo "bad arm: $ARM" >&2; exit 2 ;;
esac
LABEL="${LABEL}_K${K}"

# --- serve env --------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[ "$W4A8" = "1" ] && export VLLM_MARLIN_INPUT_DTYPE=int8
if [ "${VLLM_USE_V2_MODEL_RUNNER:-}" = "1" ]; then
  echo "REFUSE: VLLM_USE_V2_MODEL_RUNNER=1 bypasses the verified V1 path" >&2; exit 3
fi

# --- pre-flight: port must be free (wait out a prior arm's teardown) ---------
for i in $(seq 1 30); do
  curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 || break
  [ "$i" = 1 ] && echo -n ">>> port $PORT busy (prior serve tearing down) "
  echo -n .; sleep 1
  [ "$i" = 30 ] && { echo " STILL BUSY on :$PORT — kill the old serve first" >&2; exit 6; }
done
echo ""

ARGS=(vllm serve "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN"
      --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$MAXBT"
      --gpu-memory-utilization "$GMU" --port "$PORT" --trust-remote-code)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)

echo ">>> ARM=$ARM  W4A8=$W4A8  K=$K  spec=${SPEC:-<none>}"
LOG="/tmp/r04_serve_${LABEL}.log"
# new session so the whole vllm process tree (TP workers incl.) dies together
setsid bash -c 'exec "$@"' _ "${ARGS[@]}" >"$LOG" 2>&1 &
SPID=$!
cleanup(){
  kill -TERM -"$SPID" 2>/dev/null || true
  sleep 2
  kill -KILL -"$SPID" 2>/dev/null || true
  pkill -KILL -f "vllm serve $MODEL" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for readiness (fail loud if serve dies during boot) ---------------
echo -n ">>> waiting for /health "
UP=0
for i in $(seq 1 "$HEALTH_TIMEOUT"); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then echo " up (${i}s)"; UP=1; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then echo " SERVE DIED during boot — tail $LOG:"; tail -40 "$LOG"; exit 4; fi
  echo -n .; sleep 1
done
[ "$UP" = 1 ] || { echo " TIMEOUT (>${HEALTH_TIMEOUT}s); tail $LOG:"; tail -40 "$LOG"; exit 5; }

# canary: a draft-prob fallback silently erases the R04 gain
if grep -qiE "Missing cached draft prob|falling back to legacy speculative" "$LOG"; then
  echo "!!! WARNING: draft-prob fallback detected in serve log — accept-len may UNDER-read"
fi

# --- bench (OpenAI API streaming; never offline LLM()) ----------------------
BASE="http://localhost:$PORT" REP="$REP" K="$K" MAX_TOKENS="$MAXTOK" LABEL="$LABEL" \
  python3 "$HERE/api_bench_client.py"

echo ">>> done ARM=$ARM (serve log: $LOG)"
