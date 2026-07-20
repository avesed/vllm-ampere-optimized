#!/bin/bash
# Workload matrix: {K2,K3} x {greedy,prob} servers, each benched on all 4
# workloads (one server -> many prompts, no recompile). Answers: does accept_len
# / R04 gain / optimal-K vary by workload (coding vs writing vs math vs summary)?
export PATH=/home/coder/vllm-build-venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/coder/r04bench
MODEL=/home/coder/models/Qwen3.6-27B-W4A16
R=/home/coder/r04bench/results_wl.txt; : > "$R"
ts(){ date +%H:%M:%S; }
wait_gpu_free(){ for i in $(seq 1 40); do
  mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  [ "${mx:-9999}" -lt 2000 ] && return 0; sleep 2; done; }

boot_bench(){  # $1=K  $2=greedy|prob
  local K=$1 M=$2 SPEC LOG SPID
  if [ "$M" = prob ]; then
    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}"
  else
    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$K}"
  fi
  wait_gpu_free
  echo "=== [$(ts)] BOOT K=$K $M ===" >>"$R"
  LOG=/tmp/wl_serve_K${K}_${M}.log
  setsid bash -c 'exec "$@"' _ vllm serve "$MODEL" --tensor-parallel-size 2 \
    --max-model-len 34000 --max-num-seqs 16 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 --port 8000 --trust-remote-code \
    --speculative-config "$SPEC" >"$LOG" 2>&1 &
  SPID=$!
  local up=0
  for i in $(seq 1 420); do
    curl -fsS localhost:8000/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || { echo "  SERVE DIED K=$K $M" >>"$R"; tail -20 "$LOG" >>"$R"; break; }
    sleep 1
  done
  if [ "$up" = 1 ]; then
    grep -qiE "Missing cached draft prob|falling back to legacy speculative" "$LOG" \
      && echo "  !!! draft-prob fallback in K=$K $M" >>"$R"
    LABEL="K${K}-${M}" BASE=http://localhost:8000 MAX_TOKENS=512 \
      python3 /home/coder/r04bench/bench_workloads.py >>"$R" 2>&1
  fi
  kill -- -$SPID 2>/dev/null; sleep 2; kill -KILL -$SPID 2>/dev/null
  pkill -KILL -f "vllm serve $MODEL" 2>/dev/null; sleep 3
}

for K in 2 3; do
  for M in greedy prob; do
    boot_bench $K $M
  done
done
echo "ALL_DONE_WL [$(ts)]" >>"$R"
