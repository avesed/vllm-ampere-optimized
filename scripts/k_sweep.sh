#!/bin/bash
# Re-tune num_speculative_tokens under patch A. The verify is now ~flat in q, so the old
# "K=2 sweet spot (k=3 worse)" — measured when verify cost grew with K — may have flipped.
# 9B single-card, patch A on, decode tok/s (streaming) + accept-len at 16k/32k.
set -u
VBIN=/home/coder/vllm-build-venv/bin
MODEL=/home/coder/models/Qwen3.5-9B-w4a16
export PATH="$VBIN:$PATH" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FA2_KVCACHE_VERIFY=1
for K in 2 3 4; do
  echo "######## MTP num_speculative_tokens=$K (patch A on) ########"
  pkill -f "vllm serve" 2>/dev/null; sleep 4
  "$VBIN/vllm" serve "$MODEL" --tensor-parallel-size 1 --max-model-len 34000 --max-num-seqs 16 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --port 8000 --trust-remote-code \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K}" > /home/coder/serve.log 2>&1 &
  ok=0
  for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
  [ $ok = 0 ] && { echo "[K=$K serve failed]"; tail -12 /home/coder/serve.log; continue; }
  for pair in 354:16k 708:32k; do
    REP=${pair%%:*}; CTX=${pair##*:}
    REP=$REP K=$K LABEL="K$K-$CTX" BASE="http://localhost:8000" "$VBIN/python" /home/coder/api_bench_client.py
  done
  echo -n "  accept-len: "; grep -aE "Mean acceptance length" /home/coder/serve.log | grep -oE "acceptance length: [0-9.]+" | tail -3 | tr "\n" " "; echo
done
pkill -f "vllm serve" 2>/dev/null
echo "[k-sweep done]"
