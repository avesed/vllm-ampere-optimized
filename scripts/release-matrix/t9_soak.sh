#!/bin/bash
# T9 -- stability soak on the flagship serve. Gates: 0 request errors, 0 engine deaths, flat VRAM
# (a monotonic climb is the leak signal), and a degeneration canary on the outputs. Offending texts
# are retained so a canary hit can be judged rather than guessed at.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t9.log; : > "$LOG"; . ./lib.sh
MINUTES=${SOAK_MINUTES:-45}

say "T9 soak ${MINUTES}min -- 27B-W4A16 + DFlash K=7, famp on, W=16"
if serve t9-soak 8186 Qwen3.6-27B-W4A16 \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FLASHAMPERE_XQA_VERIFY=1 \
     --ARGS-- --max-num-seqs 128 \
     --speculative-config '{"method":"dflash","model":"/models/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'; then
  ( for i in $(seq 1 "$MINUTES"); do
      echo "vram_t${i}m $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')" >> "$LOG"
      sleep 60
    done ) & SAMPLER=$!
  DEGEN_FILE="${LOGDIR:-$(pwd)}/t9_degen.txt" python3 - "$MINUTES" <<'PY' >> "$LOG" 2>&1
import json, os, random, sys, time, urllib.request, concurrent.futures as cf
mins = int(sys.argv[1]); end = time.time() + mins * 60
random.seed(0); stats = {"ok": 0, "err": 0, "degen": 0}
keep = open(os.environ["DEGEN_FILE"], "w")
def one(i):
    n = random.choice([5, 40, 160, 400])
    p = ("The quick brown fox jumps over the lazy dog. " * n) + " Give one short sentence about colors."
    b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],
                    "max_tokens":random.choice([32, 96, 256]),"temperature":0.6}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8186/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        t = d["choices"][0]["message"]["content"]
        lines = [l for l in t.split("\n") if l.strip()]
        if len(lines) > 8 and len(set(lines[-8:])) == 1:
            keep.write(f"--- degen (prompt_reps={n}) ---\n{t}\n"); keep.flush()
            return "degen"
        return "ok"
    except Exception:
        return "err"
with cf.ThreadPoolExecutor(16) as ex:
    i = 0
    while time.time() < end:
        for r in ex.map(one, range(i, i + 32)): stats[r] += 1
        i += 32
print("SOAK RESULT", stats)
PY
  kill $SAMPLER 2>/dev/null
  echo "engine alive at end: $(docker ps --filter 'name=^t9-soak$' --format '{{.Names}}' | grep -cx t9-soak)" >> "$LOG"
  echo "engine deaths: $(docker logs t9-soak 2>&1 | grep -ac 'EngineDeadError\|EngineCore encountered a fatal error')" >> "$LOG"
  accept t9-soak
fi
matrix_rm t9-soak
echo -e "\nT9_DONE" >> "$LOG"
