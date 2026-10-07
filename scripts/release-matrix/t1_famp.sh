#!/bin/bash
# T1 -- famp attention legs on the flagship hd256 hybrid: the bf16cvt prefill leg must FIRE on both
# TP ranks, a mixed decode+prefill batch must stay clean (the decline path), the leg toggles must
# switch it off, and fp8 KV must fall back to a stock backend instead of failing.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t1.log; : > "$LOG"; . ./lib.sh

mixed(){ # port -- W=8 concurrent short decodes + long prefills
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request, concurrent.futures as cf
port = sys.argv[1]
def one(i):
    reps = 400 if i % 2 else 3
    p = ("The quick brown fox jumps over the lazy dog. " * reps) + " Summarize in one sentence."
    b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],
                    "max_tokens":64,"temperature":0.6}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        return d["choices"][0]["finish_reason"]
    except Exception as e:
        return f"ERR:{type(e).__name__}"
with cf.ThreadPoolExecutor(8) as ex: r = list(ex.map(one, range(32)))
print(f"  mixed W=8: {len(r)} req, errors {sum(x.startswith('ERR') for x in r)}")
PY
}
long_prompt(){ # port
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request
p = ("The quick brown fox jumps over the lazy dog. " * 220) + " What animal jumps? One word."
b = json.dumps({"model":"test","messages":[{"role":"user","content":p}],"max_tokens":200,"temperature":0.6}).encode()
d = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
print(f"  long prompt {d['usage']['prompt_tokens']} tok -> {d['choices'][0]['message']['content'][-120:]!r}")
PY
}

say "T1.1/T1.4 famp on (bf16cvt leg) -- FIRED on both ranks, mixed batch clean"
if serve t1-on 8183 Qwen3.6-27B-W4A16 -e VLLM_FLASHAMPERE=1; then
  backends t1-on; long_prompt 8183; fired t1-on
  docker logs t1-on 2>&1 | grep -a "FIRED" | grep -aoE "\(pid=[0-9]+\)|TP[0-9]|rank[0-9]" | sort -u | head >> "$LOG"
  mixed 8183; canary 8183
fi
matrix_rm t1-on

say "T1.3 leg toggles off (_PV_FP16=0 _BF16CVT=0) -> 0 FIRED"
if serve t1-off 8183 Qwen3.6-27B-W4A16 -e VLLM_FLASHAMPERE=1 -e VLLM_FLASHAMPERE_PV_FP16=0 \
     -e VLLM_FLASHAMPERE_BF16CVT=0; then
  long_prompt 8183; fired t1-off
fi
matrix_rm t1-off

say "T1.5 fp8 KV -> CUSTOM rejected, stock fallback boots coherent"
if serve t1-fp8 8183 Qwen3.6-27B-W4A16 -e VLLM_FLASHAMPERE=1 --ARGS-- --kv-cache-dtype fp8_e4m3; then
  backends t1-fp8; canary 8183
fi
matrix_rm t1-fp8
echo -e "\nT1_DONE" >> "$LOG"
