#!/bin/bash
# Profile 9B single-card MTP decode at ~16k on STOCK vllm (which HAS the torch profiler the fork
# build strips). Trace -> host dir, parsed by parse_trace.py. Runs on the HOST.
set -u
IMG=vllm/vllm-openai:v0.23.0
MH=/mnt/coder/workspaces/trevor/d2m
PORT=8011
WS=/home/trevor/workspace
PROF=$WS/prof9b; rm -rf "$PROF"; mkdir -p "$PROF"; chmod 777 "$PROF"
docker rm -f stockprof >/dev/null 2>&1; sleep 2

docker run -d --rm --name stockprof --gpus '"device=0"' -v "$MH":/work -v "$PROF":/prof \
  -e VLLM_TORCH_PROFILER_DIR=/prof "$IMG" /work/models/Qwen3.5-9B-w4a16 \
  --tensor-parallel-size 1 --max-model-len 20000 --max-num-seqs 16 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 --trust-remote-code --served-model-name m \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --port 8000 >/dev/null
for i in $(seq 1 300); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "[up]"; break; }
  docker ps --format '{{.Names}}' | grep -q stockprof || { echo "[died]"; docker logs stockprof 2>&1 | tail -30; exit 1; }
  sleep 5
done

python3 - <<'PYEOF'
import requests
BASE = "http://localhost:8011"
para = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
        "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
prompt = "请仔细阅读并总结：\n" + para * 354 + "\n请用中文系统总结。"
mid = requests.get(BASE + "/v1/models").json()["data"][0]["id"]
def req(mt):
    return requests.post(BASE + "/v1/chat/completions", json={
        "model": mid, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mt, "temperature": 0.6, "top_p": 0.95, "ignore_eos": True}, timeout=600)
req(8)
print("start_profile:", requests.post(BASE + "/start_profile", timeout=60).status_code)
req(24)
print("stop_profile:", requests.post(BASE + "/stop_profile", timeout=300).status_code)
PYEOF

echo "[flush 45s]"; sleep 45
ls -la "$PROF"
python3 "$WS/parse_trace.py" "$PROF" 2>&1 | head -30
docker rm -f stockprof >/dev/null 2>&1
echo "[done]"
