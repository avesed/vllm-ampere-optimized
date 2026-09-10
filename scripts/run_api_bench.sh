#!/bin/bash
# Run ONE server config ($1 = noMTP|MTP|int8act) x {4k,16k,32k} via the OpenAI API.
# Kept short (~one server) so it fits a single reliable SSH connection (sandbox sshd is flaky on
# long connections). Appends to apibench.log. api_bench_client.py is the streaming client.
set -u
CFG="${1:-noMTP}"
VBIN=/home/coder/vllm-build-venv/bin
MODEL="${MODEL_OVERRIDE:-/home/coder/models/Qwen3.6-27B-W4A16}"
PORT=8000
export PATH="$VBIN:$PATH" CUDA_VISIBLE_DEVICES="${CVD_OVERRIDE:-0,1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/home/coder/apibench.log
SV=""

launch() {
  "$VBIN/vllm" serve "$MODEL" --tensor-parallel-size "${TP_OVERRIDE:-2}" --max-model-len 34000 --max-num-seqs 16 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 --port "$PORT" --trust-remote-code "$@" \
    > /home/coder/serve.log 2>&1 &
  SV=$!
  for i in $(seq 1 240); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "[$CFG up]"; return 0; }
    kill -0 "$SV" 2>/dev/null || { echo "[serve died]"; tail -30 /home/coder/serve.log; return 1; }
    sleep 5
  done
  echo "[health timeout]"; return 1
}

pkill -f "vllm serve" 2>/dev/null; sleep 4
case "$CFG" in
  noMTP)   LAB=noMTP;       K=0; SPEC=();;
  MTP)     LAB=MTP;         K=2; SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}');;
  int8act) LAB=MTP_int8act; K=2; SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}'); export VLLM_MARLIN_INPUT_DTYPE=int8;;
  *) echo "unknown cfg $CFG"; exit 2;;
esac

echo "######## SERVER $LAB (K=$K) ########" | tee -a "$LOG"
if ! launch "${SPEC[@]}"; then echo "[launch FAILED: $LAB]" | tee -a "$LOG"; pkill -f "vllm serve"; exit 1; fi
for pair in 90:4k 354:16k 708:32k; do
  REP=${pair%%:*}; CTX=${pair##*:}
  REP=$REP K=$K LABEL="$LAB-$CTX" BASE="http://localhost:$PORT" \
    timeout 500 "$VBIN/python" /home/coder/api_bench_client.py 2>>"$LOG" | tee -a "$LOG"
done
pkill -f "vllm serve" 2>/dev/null; sleep 5
echo "[$CFG done]"
