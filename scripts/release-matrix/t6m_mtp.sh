#!/bin/bash
# T6m -- MTP + the spec-decode surfaces t6_spec.sh does not cover: the exactly-(K+1)-token prompt
# that once replayed a prefill through the FULL spec-verify graph (GDN state never written ->
# garbage), and the FA2 fwd_kvcache verify path at long context (FULL-graph captured; the reason
# vllm-flash-attention is pinned to a capture-safe fork commit).
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t6m.log; : > "$LOG"; . ./lib.sh

decode_ratio(){ # port -- decode tok/s at ~16k context vs ~200 tokens, streamed and timed by hand
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, sys, time, random, urllib.request
port = sys.argv[1]
def rate(reps, tag):
    random.seed(reps)
    words = [random.choice(["alpha","beta","gamma","delta","omega","sigma","kappa","theta"]) for _ in range(reps)]
    p = " ".join(words) + "\n\nWrite a long story about a lighthouse keeper."
    b = json.dumps({"model":"test","prompt":p,"max_tokens":400,"temperature":0.0,"ignore_eos":True,
                    "stream":True}).encode()
    t0 = time.time(); first = last = None; n = 0
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", b, {"Content-Type":"application/json"}), timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            if json.loads(line[5:])["choices"][0].get("text"):
                now = time.time(); first = first or now; last = now; n += 1
    # ignore_eos makes the output exactly max_tokens long; a spec-decode chunk can carry several
    # tokens, so count tokens, not chunks.
    tps = 399 / (last - first) if n > 1 else 0.0
    print(f"  {tag}: ttft {first - t0:.2f}s  decode {tps:.1f} tok/s ({n} chunks)")
    return tps
short, long_ = rate(150, "ctx~200"), rate(14000, "ctx~16k")
print(f"  long/short decode ratio: {long_ / short:.2f}  (gate >= 0.9)")
PY
}

say "T6.1 MTP 27B K=2 (FULL graphs, FA2 kvcache verify default-on)"
if serve t6m-mtp 8187 Qwen3.6-27B-W4A16 --ARGS-- --max-model-len 20000 --max-num-seqs 64 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":2}'; then
  docker logs t6m-mtp 2>&1 | grep -aE "Capturing CUDA graphs|cudagraph_mode|CUDAGraphMode" | tail -2 >> "$LOG"
  echo "  K+1 trigger (expect prompt_tokens=3, coherent continuation):" >> "$LOG"
  complete 8187 "The sky is" 40
  complete 8187 "Paris is the" 40
  canary 8187
  decode_ratio 8187
  accept t6m-mtp
fi
matrix_rm t6m-mtp

say "T6.2b same serve with VLLM_FA2_KVCACHE_VERIFY=0 (A/B: boots, ratio expected lower)"
if serve t6m-mtp-off 8187 Qwen3.6-27B-W4A16 -e VLLM_FA2_KVCACHE_VERIFY=0 --ARGS-- \
     --max-model-len 20000 --max-num-seqs 64 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":2}'; then
  decode_ratio 8187
fi
matrix_rm t6m-mtp-off

say "T6.4b DSpark K=7 exactly-8-token trigger (the original 'DSpark broken' repro) + canary"
if serve t6m-dspark 8187 Qwen3.6-27B-W4A16 --ARGS-- --max-num-seqs 64 \
     --speculative-config "{\"method\":\"dspark\",\"model\":\"/models/${DSPARK_HEAD:-dspark_27b_published}\",\"num_speculative_tokens\":7}"; then
  complete 8187 "请用中文写一句关于春天的话。" 60
  canary 8187
  accept t6m-dspark
fi
matrix_rm t6m-dspark
echo -e "\nT6M_DONE" >> "$LOG"
