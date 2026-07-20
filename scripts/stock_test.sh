#!/bin/bash
# Run STOCK vllm/vllm-openai:v0.23.0 (single card) noMTP + MTP x {4k,16k,32k} on the 9B-W4A16, to
# isolate whether the long-context MTP degradation is fork-specific or upstream-vLLM behavior.
# Runs ON THE HOST (stable sshd); client uses host python3.
set -u
IMG=vllm/vllm-openai:v0.23.0
MH=/mnt/coder/workspaces/trevor/d2m          # host path of the coder container's /home/coder
PORT=8011
WS=/home/trevor/workspace
LOG=$WS/stock_apibench.log; : > "$LOG"

run_cfg() {  # $1 label, $2 K, rest = extra serve flags
  local LAB=$1 K=$2; shift 2
  docker rm -f stockvllm >/dev/null 2>&1; sleep 3
  echo "######## STOCK $LAB ########" | tee -a "$LOG"
  docker run -d --rm --name stockvllm --gpus '"device=0"' -v "$MH":/work -p $PORT:8000 \
    "$IMG" /work/models/Qwen3.5-9B-w4a16 --tensor-parallel-size 1 --max-model-len 34000 \
    --max-num-seqs 16 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 \
    --trust-remote-code --served-model-name m "$@" --port 8000 >/dev/null
  local i
  for i in $(seq 1 300); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "[$LAB up]" | tee -a "$LOG"; break; }
    docker ps --format '{{.Names}}' | grep -q stockvllm || { echo "[died $LAB]" | tee -a "$LOG"; docker logs stockvllm 2>&1 | tail -35; return; }
    sleep 5
  done
  for pair in 90:4k 354:16k 708:32k; do
    REP=${pair%%:*}; CTX=${pair##*:}
    REP=$REP K=$K LABEL="$LAB-$CTX" BASE="http://localhost:$PORT" python3 "$WS/api_bench_client.py" 2>>"$LOG" | tee -a "$LOG"
  done
  docker logs stockvllm 2>&1 | grep -aE "Mean acceptance length" | tail -2 | tee -a "$LOG"
  docker rm -f stockvllm >/dev/null 2>&1
}

run_cfg stock-noMTP 0
run_cfg stock-MTP 2 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
echo "===== STOCK SUMMARY =====" | tee -a "$LOG"; grep RESULT "$LOG"
