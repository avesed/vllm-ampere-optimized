#!/bin/bash
# Lossless lever: --async-scheduling overlaps the K MTP-head forwards + num_accepted GPU->CPU sync
# behind the GPU step (the #35387 fix path). 9B TP1 (TP2 has the #41190 async crash risk), K=2,
# patch A on. Compare decode @16k/32k vs sync baseline (K-sweep K=2 = 96.3 / 87.9).
set -u
VBIN=/home/coder/vllm-build-venv/bin
MODEL=/home/coder/models/Qwen3.5-9B-w4a16
export PATH="$VBIN:$PATH" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FA2_KVCACHE_VERIFY=1
for MODE in async sync; do
  echo "######## MTP K=2 scheduling=$MODE ########"
  pkill -f "vllm serve" 2>/dev/null; sleep 4
  EXTRA=""; [ "$MODE" = async ] && EXTRA="--async-scheduling"
  "$VBIN/vllm" serve "$MODEL" --tensor-parallel-size 1 --max-model-len 34000 --max-num-seqs 16 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --port 8000 --trust-remote-code $EXTRA \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' > /home/coder/serve.log 2>&1 &
  ok=0; for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
  [ $ok = 0 ] && { echo "[$MODE serve failed]"; tail -15 /home/coder/serve.log; continue; }
  for pair in 354:16k 708:32k; do
    REP=${pair%%:*}; CTX=${pair##*:}
    REP=$REP K=2 LABEL="$MODE-$CTX" BASE="http://localhost:8000" "$VBIN/python" /home/coder/api_bench_client.py
  done
  echo -n "  accept-len: "; grep -aE "Mean acceptance length" /home/coder/serve.log | grep -oE "acceptance length: [0-9.]+" | tail -2 | tr "\n" " "; echo
done
pkill -f "vllm serve" 2>/dev/null; echo "[async test done]"
