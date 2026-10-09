#!/bin/bash
# T6g -- draft-implementation A/B under GREEDY decoding. With temp 0 the target's output is fixed,
# so acceptance length isolates the draft path (e.g. a rebuilt qwen3_dflash.py) from sampling
# noise; the 6-prompt temp-0.6 battery in t6_spec moves +-0.2 between runs. Run on both images and
# compare acceptance and the output digest (equal digests = identical target greedy outputs).
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t6g.log; : > "$LOG"; . ./lib.sh

greedy(){ # port
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import hashlib, json, re, sys, urllib.request, concurrent.futures as cf
port = sys.argv[1]; base = f"http://127.0.0.1:{port}"
qs = ["If a train travels 60 km in 45 minutes, what is its speed in km/h?",
      "What is 17 times 23? Show your work.", "A rectangle is 7 by 9. What is its perimeter and area?",
      "Sarah has 48 apples and gives away a third. How many are left?",
      "Write a Python function that returns the n-th Fibonacci number iteratively.",
      "Write a Python function that checks whether a string is a palindrome.",
      "Explain in a short paragraph why the sky is blue.",
      "What are the main differences between TCP and UDP?",
      "用三句话解释 Transformer 里的自注意力机制。", "简要介绍一下长城的历史。",
      "Give three tips for writing clear technical documentation.",
      "Solve for x: 3x + 7 = 25. Explain each step."]
def counters():
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for key in ("num_drafts", "num_accepted_tokens"):
        pat = re.compile(rf"^vllm:spec_decode_{key}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)$", re.M)
        out[key] = sum(float(v) for v in pat.findall(txt))
    return out
def ask(q):
    b = json.dumps({"model":"test","messages":[{"role":"user","content":q}],
                    "max_tokens":512,"temperature":0.0}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=600))
    return d["choices"][0]["message"]["content"] or ""
c0 = counters()
with cf.ThreadPoolExecutor(4) as ex: outs = list(ex.map(ask, qs))
c1 = counters()
dr = c1["num_drafts"] - c0["num_drafts"]; ac = c1["num_accepted_tokens"] - c0["num_accepted_tokens"]
digest = hashlib.sha256("\x00".join(outs).encode()).hexdigest()[:16]
print(f"  greedy accept {1 + ac / dr:.3f} over {int(dr)} drafts | output digest {digest}")
PY
}

for cfg in "dflash Qwen3.6-27B-DFlash" "dspark ${DSPARK_HEAD:-dspark_27b_published}"; do
  set -- $cfg
  say "T6g $1 K=7 greedy ($2)"
  if serve "t6g-$1" 8191 Qwen3.6-27B-W4A16 --ARGS-- --max-num-seqs 64 \
       --speculative-config "{\"method\":\"$1\",\"model\":\"/models/$2\",\"num_speculative_tokens\":7}"; then
    greedy 8191
  fi
  matrix_rm "t6g-$1"
done
echo -e "\nT6G_DONE" >> "$LOG"
