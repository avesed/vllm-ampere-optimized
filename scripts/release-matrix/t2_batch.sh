#!/bin/bash
# T2 -- de-taxed batch prefill (VLLM_FAMP_BATCH_PREFILL=1). The leg needs fp16 query AND fp16 KV,
# and every staged checkpoint is bf16, so --dtype float16 is forced: without it the leg silently
# never fires and the block "passes" having exercised nothing.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t2.log; : > "$LOG"; . ./lib.sh

ask(){ python3 - "$1" "$2" "$3" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request
port, rep, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = ("The quick brown fox jumps over the lazy dog. " * rep) + " Name one primary color."
b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],
                "max_tokens":40,"temperature":0.0}).encode()
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
    print(f"  {tag}: prompt_tokens={d['usage']['prompt_tokens']} finish={d['choices'][0]['finish_reason']}")
except Exception as e:
    print(f"  {tag}: REQUEST FAILED {type(e).__name__}: {str(e)[:110]}")
PY
}

say "T2.1/T2.3 batch prefill ON, fp16-served -- gate boundary (FAMP_BP_MIN_TOKENS=512)"
if serve t2-on 8184 Qwen3.6-27B-W4A16 \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FAMP_BATCH_PREFILL=1 \
     --ARGS-- --dtype float16; then
  ask 8184 40  "short ~416tok (must stay on the per-request leg)"
  ask 8184 200 "long ~2016tok (batch path expected)"
  fired t2-on
fi
docker rm -f t2-on >/dev/null 2>&1

say "T2.5/T2.6 FAMP_BP_NOZERO=1 + concurrency W=8"
if serve t2-nozero 8184 Qwen3.6-27B-W4A16 \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FAMP_BATCH_PREFILL=1 -e FAMP_BP_NOZERO=1 \
     --ARGS-- --dtype float16; then
  ask 8184 200 "long, NOZERO=1"
  python3 - <<'PY' >> "$LOG" 2>&1
import json, urllib.request, concurrent.futures as cf
def one(i):
    p = ("The quick brown fox jumps over the lazy dog. " * (5 + (i % 40) * 10)) + " Name one color."
    b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],
                    "max_tokens":32,"temperature":0.6}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8184/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        return d["choices"][0]["finish_reason"]
    except Exception as e:
        return f"ERR:{type(e).__name__}"
with cf.ThreadPoolExecutor(8) as ex: r = list(ex.map(one, range(32)))
e = [x for x in r if x.startswith("ERR")]
print(f"  concurrency W=8: {len(r)} requests, {len(e)} errors {e[:3]}")
PY
  fired t2-nozero
fi
docker rm -f t2-nozero >/dev/null 2>&1
echo -e "\nT2_DONE" >> "$LOG"
