#!/bin/bash
# T6 -- spec decode. One fixed protocol (6 arithmetic prompts, temp 0.6, K=7). Acceptance length is
# per-category and per-sampling-regime, so these numbers are only comparable to the SAME battery on
# another image -- never to a band quoted from a different protocol.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t6.log; : > "$LOG"; . ./lib.sh

spec(){ # name port draft K method [extra env...]
  local name=$1 port=$2 draft=$3 K=$4 method=$5; shift 5
  serve "$name" "$port" Qwen3.6-27B-W4A16 "$@" --ARGS-- \
    --max-num-seqs 128 \
    --speculative-config "{\"method\":\"$method\",\"model\":\"/models/$draft\",\"num_speculative_tokens\":$K}"
}
battery(){ python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request
port = sys.argv[1]
qs = [("If a train travels 60 km in 45 minutes, what is its speed in km/h?", "80"),
      ("A shop sells pens at 3 for $5. How much do 12 pens cost?", "20"),
      ("What is 17 times 23?", "391"),
      ("Sarah has 48 apples and gives away a third. How many are left?", "32"),
      ("A rectangle is 7 by 9. What is its perimeter?", "32"),
      ("If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?", "5")]
bad = wrong = 0
for q, want in qs:
    b = json.dumps({"model":"test","messages":[{"role":"user","content":q}],
                    "max_tokens":1500,"temperature":0.6}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        t = d["choices"][0]["message"]["content"]
        lines = [l for l in t.split("\n") if l.strip()]
        if len(lines) > 8 and len(set(lines[-8:])) == 1: bad += 1
        if want not in t[-400:]: wrong += 1
    except Exception as e:
        bad += 1; print("  REQUEST FAILED:", type(e).__name__, str(e)[:100])
print(f"  degenerate: {bad}/6   wrong-answer: {wrong}/6")
PY
}

say "T6.5 DFlash 27B K=7"
if spec t6-dflash 8189 Qwen3.6-27B-DFlash 7 dflash \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FLASHAMPERE_XQA_VERIFY=1; then
  battery 8189; accept t6-dflash; fi
matrix_rm t6-dflash

say "T6.4 DSpark 27B K=7"
if spec t6-dspark 8189 dspark_27b_published 7 dspark \
     -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FLASHAMPERE_XQA_VERIFY=1; then
  battery 8189; accept t6-dspark; fi
matrix_rm t6-dspark
echo -e "\nT6_DONE" >> "$LOG"
