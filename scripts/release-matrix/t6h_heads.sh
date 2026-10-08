#!/bin/bash
# T6h -- draft-head A/B: the published DSpark head vs its repaired copy (embed_tokens holes filled
# from the target), and cross-generation use (Qwen3.6 head on a Qwen3.8 target and vice versa)
# against RadixArk's Qwen3.8 head and Qwen3.8's own MTP. Acceptance is reported PER CATEGORY from
# /metrics counter deltas (acceptance is per-category and per-sampling-regime; one protocol here:
# temp 0.6, top_p 0.95, 1024 tokens) -- compare only within this block.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t6h.log; : > "$LOG"; . ./lib.sh

battery(){ # port
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, re, sys, urllib.request, concurrent.futures as cf
port = sys.argv[1]
base = f"http://127.0.0.1:{port}"
cats = {
  "math": ["If a train travels 60 km in 45 minutes, what is its speed in km/h?",
           "A shop sells pens at 3 for $5. How much do 12 pens cost?",
           "What is 17 times 23?", "Sarah has 48 apples and gives away a third. How many are left?",
           "A rectangle is 7 by 9. What is its perimeter?",
           "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?"],
  "zh": ["用三句话解释 Transformer 里的自注意力机制。", "写一首四行的小诗，主题是秋天。",
         "列出三种常见的排序算法并简单比较。", "简要介绍一下长城的历史。"],
  "code": ["Write a Python function that returns the n-th Fibonacci number iteratively.",
           "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.",
           "Write a bash one-liner that counts lines in all .py files under the current directory."],
  "chat": ["Explain in a short paragraph why the sky is blue.",
           "Give three tips for writing clear technical documentation.",
           "What are the main differences between TCP and UDP?"],
}
def counters():
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        pat = re.compile(rf"^vllm:spec_decode_{key}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)$", re.M)
        out[key] = sum(float(v) for v in pat.findall(txt))
    return out
def ask(q):
    b = json.dumps({"model":"test","messages":[{"role":"user","content":q}],
                    "max_tokens":1024,"temperature":0.6,"top_p":0.95}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            base + "/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=600))
        return d["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return f"ERR {type(e).__name__}"
row = []
for cat, qs in cats.items():
    c0 = counters()
    with cf.ThreadPoolExecutor(4) as ex: outs = list(ex.map(ask, qs))
    c1 = counters()
    dr = c1["num_drafts"] - c0["num_drafts"]; ac = c1["num_accepted_tokens"] - c0["num_accepted_tokens"]
    errs = sum(o.startswith("ERR") for o in outs)
    row.append(f"{cat} {1 + ac / dr:.2f}" if dr else f"{cat} n/a")
    if errs: row.append(f"({cat} errors {errs})")
print("  accept per category:", " | ".join(row))
PY
}

pair(){ # name target draft-config-json [extra docker args...]
  local name=$1 target=$2 spec=$3; shift 3
  say "$name: target=$target spec=$spec"
  if serve "$name" 8188 "$target" "$@" --ARGS-- --max-num-seqs 64 --speculative-config "$spec"; then
    docker logs "$name" 2>&1 | grep -aE "embed_tokens|Sharing target|separate embedding|lm_head" | sort -u | head -3 >> "$LOG"
    battery 8188; canary 8188
  fi
  matrix_rm "$name"
}
ds(){ echo "{\"method\":\"dspark\",\"model\":\"/models/$1\",\"num_speculative_tokens\":7}"; }

pair t6h-36-pub   Qwen3.6-27B-W4A16 "$(ds dspark_27b_published)"
pair t6h-36-rep   Qwen3.6-27B-W4A16 "$(ds dspark_27b_repaired)"
pair t6h-38-rep   Qwen3.8-27B-W4A16 "$(ds dspark_27b_repaired)"
pair t6h-38-radix Qwen3.8-27B-W4A16 "$(ds radixark_38_dspark)"
pair t6h-36-radix Qwen3.6-27B-W4A16 "$(ds radixark_38_dspark)"
pair t6h-38-mtp   Qwen3.8-27B-W4A16 '{"method":"mtp","num_speculative_tokens":2}'
echo -e "\nT6H_DONE" >> "$LOG"
