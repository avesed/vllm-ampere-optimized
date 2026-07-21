#!/bin/bash
# Profile 9B single-card MTP decode at ~16k to name the long-context kernel. Robust trace capture:
# long flush wait + broad find + serve.log profiler messages + don't kill until parsed.
set -u
VBIN=/home/coder/vllm-build-venv/bin
MODEL=/home/coder/models/Qwen3.5-9B-w4a16
PORT=8000
export PATH="$VBIN:$PATH" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_TORCH_PROFILER_DIR=/home/coder/prof9b
rm -rf "$VLLM_TORCH_PROFILER_DIR"; mkdir -p "$VLLM_TORCH_PROFILER_DIR"
pkill -f "vllm serve" 2>/dev/null; sleep 4

"$VBIN/vllm" serve "$MODEL" --tensor-parallel-size 1 --max-model-len 20000 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --port "$PORT" --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' > /home/coder/serve.log 2>&1 &
SV=$!
for i in $(seq 1 240); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "[up]"; break; }
  kill -0 "$SV" 2>/dev/null || { echo "[died]"; tail -25 /home/coder/serve.log; exit 1; }
  sleep 5
done

"$VBIN/python" - <<'PYEOF'
import requests
BASE = "http://localhost:8000"
para = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
        "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
prompt = "请仔细阅读并总结：\n" + para * 354 + "\n请用中文系统总结。"
mid = requests.get(BASE + "/v1/models").json()["data"][0]["id"]
def req(mt):
    return requests.post(BASE + "/v1/chat/completions", json={
        "model": mid, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mt, "temperature": 0.6, "top_p": 0.95, "ignore_eos": True}, timeout=600)
req(8)
requests.post(BASE + "/start_profile", timeout=60)
req(24)
requests.post(BASE + "/stop_profile", timeout=300)
print("[profiled 24 decode steps @16k]")
PYEOF

echo "[flushing 40s]"; sleep 40
echo "=== serve.log profiler messages ==="
grep -aiE "profile|trace|chrome" /home/coder/serve.log | tail -8 | cut -c1-160
echo "=== trace files (broad) ==="
find /home/coder/prof9b /tmp /home/coder -maxdepth 4 -name "*.json*" -newermt "-5 min" 2>/dev/null | grep -iE "trace|prof" | head
ls -la /home/coder/prof9b/ 2>/dev/null
echo "=== parse ==="
"$VBIN/python" /home/coder/parse_trace.py /home/coder/prof9b 2>&1 | head -30
pkill -f "vllm serve" 2>/dev/null; sleep 3
echo "[done]"
