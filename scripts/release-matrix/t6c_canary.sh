#!/bin/bash
# T6c -- is a canary hit under spec decode the target's own sampling, or the spec path? Spec decode
# is lossless, so with the same target the degenerate rate must match the no-spec rate. Repeats the
# canary prompts N times per config at temp 0.6 and reports degenerate counts side by side.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t6c.log; : > "$LOG"; . ./lib.sh
TARGET=${CANARY_TARGET:-Qwen3.8-27B-W4A16}; HEAD=${CANARY_HEAD:-dspark_27b_repaired}; REPS=${CANARY_REPS:-4}

repeat_canary(){ # port
  python3 - "$1" "$REPS" <<'PY' >> "$LOG" 2>&1
import json, re, sys, urllib.request, concurrent.futures as cf
port, reps = sys.argv[1], int(sys.argv[2])
qs = ["用三句话解释 Transformer 里的自注意力机制。", "请用中文写一句关于春天的话。",
      "把这句话翻译成英文：床前明月光，疑是地上霜。", "列出三种常见的排序算法并简单比较。",
      "Explain in two sentences why the sky is blue.", "写一首四行的小诗，主题是秋天。",
      "What is the capital of Australia, and why is it not Sydney?", "用一句话介绍一下长城。"]
def one(q):
    b = json.dumps({"model":"test","messages":[{"role":"user","content":q}],
                    "max_tokens":300,"temperature":0.6,"top_p":0.95}).encode()
    t = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
    t = t["choices"][0]["message"]["content"] or ""
    return bool("�" in t or re.search(r"(.{4,}?)\1{5,}", t, re.S)), t
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(one, qs * reps))
hits = [t for h, t in res if h]
print(f"  degenerate {len(hits)}/{len(res)}")
for t in hits[:3]: print("   ", repr(t[-200:]))
PY
}

say "T6c $TARGET no spec"
if serve t6c-nospec 8192 "$TARGET" --ARGS-- --max-num-seqs 64; then repeat_canary 8192; fi
matrix_rm t6c-nospec
say "T6c $TARGET + DSpark $HEAD"
if serve t6c-spec 8192 "$TARGET" --ARGS-- --max-num-seqs 64 \
     --speculative-config "{\"method\":\"dspark\",\"model\":\"/models/$HEAD\",\"num_speculative_tokens\":7}"; then
  repeat_canary 8192; fi
matrix_rm t6c-spec
echo -e "\nT6C_DONE" >> "$LOG"
