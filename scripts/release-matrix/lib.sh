# Shared helpers for the release matrix blocks.
# Env: IMG (image under test), MODELS (staged checkpoint dir), LOGDIR.
# Design notes learned the hard way:
#   - never mount patched files here: the block must prove the IMAGE is correct;
#   - always emit an explicit TIMEOUT marker -- a container that lives but never serves is
#     otherwise indistinguishable from "still booting" to a log watcher;
#   - capture the TAIL of a traceback, never the head (the head is always boilerplate);
#   - NEVER clean up by name. `docker ps --filter name=X` is a SUBSTRING match, so a short or
#     careless X sweeps unrelated containers -- `--filter name=t` once removed cf-tunnel,
#     coder-database-1, gpu-hot and arc-desktop off a production host. Every container this
#     matrix starts carries MATRIX_LABEL, and cleanup only ever selects on that label.
: "${IMG:?set IMG}"; : "${MODELS:?set MODELS}"; : "${LOGDIR:=$(pwd)}"
MATRIX_LABEL="vllm-release-matrix=1"

# Remove ONLY containers this matrix created. Selects on the label, never on a name.
matrix_cleanup(){
  local ids; ids=$(docker ps -aq --filter "label=$MATRIX_LABEL")
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  return 0
}

# Remove one container, but only if it carries our label -- so a typo'd name is a no-op
# instead of an outage.
matrix_rm(){
  local c=$1
  if [ "$(docker inspect -f '{{index .Config.Labels "vllm-release-matrix"}}' "$c" 2>/dev/null)" = "1" ]; then
    docker rm -f "$c" >/dev/null 2>&1
  fi
  return 0
}

say(){ echo -e "\n########## $*" >> "$LOG"; }

serve(){ # name port model [extra docker args...] [--ARGS-- extra server args...]
  local name=$1 port=$2 model=$3; shift 3
  local denv=() sargs=()
  while [ $# -gt 0 ] && [ "$1" != "--ARGS--" ]; do denv+=("$1"); shift; done
  [ $# -gt 0 ] && shift; sargs=("$@")
  matrix_rm "$name"
  docker run -d --name "$name" --label "$MATRIX_LABEL" --gpus all --ipc=host --shm-size=8g \
    -v "$MODELS":/models:ro -p "$port":8000 "${denv[@]}" "$IMG" \
    --model "/models/$model" --served-model-name test \
    --tensor-parallel-size 2 --disable-custom-all-reduce --max-model-len 8192 \
    --gpu-memory-utilization 0.85 "${sargs[@]}" >/dev/null 2>&1
  for i in $(seq 1 95); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "READY ${i}0s" >> "$LOG"; return 0; }
    docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -qx "$name" || {
      echo "DIED at ${i}0s" >> "$LOG"
      docker logs "$name" 2>&1 | grep -aiE "Error|Exception|assert|raise " | tail -6 >> "$LOG"
      return 1; }
    sleep 10
  done
  echo "TIMEOUT (container alive but never served)" >> "$LOG"; return 1
}

# GSM8K over the API with one fixed protocol (sampled -- never greedy on these thinking models,
# 0.6/0.95/20, \boxed{} answers). Only ever compare a number to the same call on another image.
gsm8k(){ # port tag [n]
  GSM8K_JSONL="${GSM8K_JSONL:?set GSM8K_JSONL}" python3 ../../eval/gsm8k_api_eval.py \
    --base "http://127.0.0.1:$1" --tag "$2" --n "${3:-40}" --temperature 0.6 --top-p 0.95 \
    --top-k 20 --max-tokens 6000 --concurrency 20 --boxed 2>&1 | grep -aE "RESULT|thinking spans|mean completion" >> "$LOG"
}

# Fire one raw /v1/completions prompt and print the text (for exact-token-count triggers).
complete(){ # port prompt max_tokens
  python3 - "$1" "$2" "$3" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request
port, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = json.dumps({"model":"test","prompt":prompt,"max_tokens":n,"temperature":0.0}).encode()
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", b, {"Content-Type":"application/json"}), timeout=300))
    print(f"  prompt_tokens={d['usage']['prompt_tokens']} -> {d['choices'][0]['text'][:200]!r}")
except Exception as e:
    print(f"  REQUEST FAILED {type(e).__name__}: {str(e)[:120]}")
PY
}

# zh+en garble canary: 8 chat prompts, flag looping/repeated spans or U+FFFD. Texts of hits are
# printed so a hit can be judged.
canary(){ # port
  python3 - "$1" <<'PY' >> "$LOG" 2>&1
import json, re, sys, urllib.request
port = sys.argv[1]
qs = ["用三句话解释 Transformer 里的自注意力机制。", "请用中文写一句关于春天的话。",
      "把这句话翻译成英文：床前明月光，疑是地上霜。", "列出三种常见的排序算法并简单比较。",
      "Explain in two sentences why the sky is blue.", "写一首四行的小诗，主题是秋天。",
      "What is the capital of Australia, and why is it not Sydney?", "用一句话介绍一下长城。"]
hits = 0
for q in qs:
    b = json.dumps({"model":"test","messages":[{"role":"user","content":q}],
                    "max_tokens":300,"temperature":0.6,"top_p":0.95}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", b, {"Content-Type":"application/json"}), timeout=300))
        t = d["choices"][0]["message"]["content"] or ""
    except Exception as e:
        hits += 1; print("  REQUEST FAILED:", type(e).__name__, str(e)[:100]); continue
    if "�" in t or re.search(r"(.{4,}?)\1{5,}", t, re.S):
        hits += 1; print("  CANARY HIT:", repr(t[-300:]))
print(f"  canary: {hits}/{len(qs)} degenerate")
PY
}

fired(){ echo "  famp fired: $(docker logs "$1" 2>&1 | grep -ac 'FLASHAMPERE .* FIRED')" >> "$LOG"
         docker logs "$1" 2>&1 | grep -a 'FLASHAMPERE .* FIRED' | tail -2 >> "$LOG"; }
backends(){ docker logs "$1" 2>&1 | grep -aE "Using .*attention backend" | grep -avi all-reduce | sort -u >> "$LOG"; }
accept(){ docker logs "$1" 2>&1 | grep -a "acceptance length" | tail -1 >> "$LOG"; }
