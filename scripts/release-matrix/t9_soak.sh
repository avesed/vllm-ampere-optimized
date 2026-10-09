#!/bin/bash
# T9 -- stability soak on the flagship serves. Gates: 0 request errors, 0 engine deaths, flat VRAM
# (a monotonic climb is the leak signal), 0 degenerate outputs and 0 ABNORMAL STARTS. Offending
# texts are retained so a hit can be judged rather than guessed at.
#
# Abnormal start = an output that does not open with the target's thinking preamble ("Here's a
# thinking process" / "Thinking Process:" on Qwen3.6). It is ~10x more sensitive than the
# 8-identical-lines canary: the 0.31 async accepted-count race (upstream #51571) showed up as 0.9%
# abnormal starts ("Here for's for 9 the") but only 0.16% fully degenerate outputs. Expected rate
# with spec decode off, or on a healthy spec path, is 0 in thousands.
#
# Two serves: the flagship DFlash K=7 with flashampere on, and the default env (flashampere off)
# with MTP K=2 -- the README's recommended config. Both keep prefix caching and async scheduling on
# (the defaults), with identical prompts recurring so prefix-cache hits are constant.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t9.log; : > "$LOG"; . ./lib.sh
MINUTES=${SOAK_MINUTES:-30}

soak(){ # name minutes -- runs the load against :8186 and writes t9_<name>.jsonl
  local name=$1 mins=$2
  ( for i in $(seq 1 "$mins"); do
      echo "vram_t${i}m $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')" >> "$LOG"
      sleep 60
    done ) & local sampler=$!
  OUT="${LOGDIR:-$(pwd)}/t9_${name}.jsonl" python3 - "$mins" <<'PY' >> "$LOG" 2>&1
import json, os, random, sys, time, urllib.request, concurrent.futures as cf
mins = int(sys.argv[1]); end = time.time() + mins * 60
random.seed(0); stats = {"ok": 0, "err": 0, "degen": 0, "abnormal": 0}
out = open(os.environ["OUT"], "w")
def one(i):
    n = random.choice([5, 40, 160, 400]); mt = random.choice([32, 96, 256])
    p = ("The quick brown fox jumps over the lazy dog. " * n) + " Give one short sentence about colors."
    b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],
                    "max_tokens":mt,"temperature":0.6}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8186/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        t = d["choices"][0]["message"]["content"] or ""
    except Exception:
        return "err"
    lines = [l for l in t.split("\n") if l.strip()]
    degen = len(lines) > 8 and len(set(lines[-8:])) == 1
    abnormal = not (t.startswith("Here's a thinking") or t.startswith("Thinking Process"))
    out.write(json.dumps({"reps": n, "max_tokens": mt, "degen": degen, "abnormal": abnormal,
                          "head": t[:80], "tail": t[-120:]}) + "\n"); out.flush()
    return "degen" if degen else "abnormal" if abnormal else "ok"
with cf.ThreadPoolExecutor(16) as ex:
    i = 0
    while time.time() < end:
        for r in ex.map(one, range(i, i + 32)): stats[r] += 1
        i += 32
print("SOAK RESULT", stats)
PY
  kill "$sampler" 2>/dev/null
  python3 - "${LOGDIR:-$(pwd)}/t9_${name}.jsonl" <<'PY' >> "$LOG" 2>&1
import json, sys
bad = [r for r in map(json.loads, open(sys.argv[1])) if r["abnormal"] or r["degen"]]
for r in bad[:5]: print("   ", repr(r["head"][:60]))
PY
}

say "T9.1 soak ${MINUTES}min -- 27B-W4A16 + DFlash K=7, famp on, W=16"
if serve t9-soak 8186 Qwen3.6-27B-W4A16 \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FLASHAMPERE_XQA_VERIFY=1 \
     --ARGS-- --max-num-seqs 128 \
     --speculative-config '{"method":"dflash","model":"/models/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'; then
  soak dflash "$MINUTES"
  echo "engine alive at end: $(docker ps --filter 'name=^t9-soak$' --format '{{.Names}}' | grep -cx t9-soak)" >> "$LOG"
  echo "engine deaths: $(docker logs t9-soak 2>&1 | grep -ac 'EngineDeadError\|EngineCore encountered a fatal error')" >> "$LOG"
  accept t9-soak
fi
matrix_rm t9-soak

say "T9.2 soak ${MINUTES}min -- 27B-W4A16 + MTP K=2, default env (famp off), W=16"
if serve t9-soak 8186 Qwen3.6-27B-W4A16 --ARGS-- --max-num-seqs 128 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":2}'; then
  soak mtp "$MINUTES"
  echo "engine alive at end: $(docker ps --filter 'name=^t9-soak$' --format '{{.Names}}' | grep -cx t9-soak)" >> "$LOG"
  echo "engine deaths: $(docker logs t9-soak 2>&1 | grep -ac 'EngineDeadError\|EngineCore encountered a fatal error')" >> "$LOG"
  accept t9-soak
fi
matrix_rm t9-soak
echo -e "\nT9_DONE" >> "$LOG"
