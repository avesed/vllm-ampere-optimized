#!/bin/bash
# Coherence smoke: 27B tp2 (instruct, real deployment) with the verify patch ON — confirm zh+en
# generations are sensible (not garbage from a wrong verify or tp2 shm corruption).
set -u
VBIN=/home/coder/vllm-build-venv/bin
export PATH="$VBIN:$PATH" CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FA2_KVCACHE_VERIFY=1
pkill -f "vllm serve" 2>/dev/null; sleep 4
"$VBIN/vllm" serve /home/coder/models/Qwen3.6-27B-W4A16 --tensor-parallel-size 2 --max-model-len 20000 \
  --max-num-seqs 16 --gpu-memory-utilization 0.90 --port 8000 --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' > /home/coder/serve.log 2>&1 &
for i in $(seq 1 300); do curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "[up]"; break; }; sleep 5; done
python3 - <<'PY'
import requests
B = "http://localhost:8000"
m = requests.get(B + "/v1/models").json()["data"][0]["id"]
for q in ["用三句话解释 Transformer 里的自注意力机制是怎么工作的。",
          "Explain in 3 sentences why speculative decoding can speed up LLM inference.",
          "请把这句话翻译成英文：床前明月光，疑是地上霜。"]:
    r = requests.post(B + "/v1/chat/completions", json={
        "model": m, "messages": [{"role": "user", "content": q}],
        "max_tokens": 160, "temperature": 0.6, "top_p": 0.95}, timeout=120).json()
    print("Q:", q)
    print("A:", r["choices"][0]["message"]["content"].strip()[:400])
    print("-" * 60)
PY
pkill -f "vllm serve" 2>/dev/null
echo "[coherence done]"
