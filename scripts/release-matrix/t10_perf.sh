#!/bin/bash
# T10 -- perf regression gate. Run on two images back to back on an otherwise idle box, and only
# compare medians from the same block: one sample per point (as T6m takes) is noise at the +-8%
# level. Decode is timed by hand from the stream (tokens = max_tokens via ignore_eos); prefill is
# prompt_tokens / TTFT with prefix caching off and a fresh random prompt per request. max-num-seqs 64:
# at 20k context + MTP the 27B hybrid has only ~228 Mamba blocks, below the default 256.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t10.log; : > "$LOG"; . ./lib.sh
REPS=${PERF_REPS:-5}

bench(){ # port mode(decode|prefill) words max_tokens
  python3 - "$1" "$2" "$3" "$4" "$REPS" <<'PY' >> "$LOG" 2>&1
import json, random, statistics, sys, time, urllib.request
port, mode, words, max_tokens, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
vocab = ["alpha","beta","gamma","delta","omega","sigma","kappa","theta","river","stone","cloud","ember"]
vals = []
for r in range(reps + 1):  # first request warms up and is discarded
    random.seed(time.time_ns())
    p = " ".join(random.choice(vocab) for _ in range(words)) + "\n\nWrite a long story about a lighthouse keeper."
    b = json.dumps({"model":"test","prompt":p,"max_tokens":max_tokens,"temperature":0.0,"ignore_eos":True,
                    "stream":True,"stream_options":{"include_usage":True}}).encode()
    t0 = time.time(); first = last = None; ptoks = None
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", b, {"Content-Type":"application/json"}), timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): ptoks = d["usage"]["prompt_tokens"]
            if d.get("choices") and d["choices"][0].get("text"):
                now = time.time(); first = first or now; last = now
    if r == 0: continue
    vals.append((max_tokens - 1) / (last - first) if mode == "decode" else ptoks / (first - t0))
print(f"  {mode} words={words} max_tokens={max_tokens}: median {statistics.median(vals):.1f} tok/s "
      f"(min {min(vals):.1f} max {max(vals):.1f}, n={len(vals)})")
PY
}

say "T10.1 27B W4A16, no spec: decode (ctx ~200) + prefill (~8k, prefix cache off)"
if serve t10-plain 8190 Qwen3.6-27B-W4A16 --ARGS-- --max-model-len 20000 --max-num-seqs 64 \
     --no-enable-prefix-caching; then
  bench 8190 decode 150 400; bench 8190 prefill 7000 8
fi
matrix_rm t10-plain

say "T10.2 27B W4A16 + MTP K=2: decode at ctx ~200 and ~16k (FA2 kvcache verify on)"
if serve t10-mtp 8190 Qwen3.6-27B-W4A16 --ARGS-- --max-model-len 20000 --max-num-seqs 64 \
     --no-enable-prefix-caching \
     --speculative-config '{"method":"mtp","num_speculative_tokens":2}'; then
  bench 8190 decode 150 400; bench 8190 decode 14000 400
fi
matrix_rm t10-mtp
echo -e "\nT10_DONE" >> "$LOG"
