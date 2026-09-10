# Shared helpers for the release matrix blocks.
# Env: IMG (image under test), MODELS (staged checkpoint dir), LOGDIR.
# Design notes learned the hard way:
#   - never mount patched files here: the block must prove the IMAGE is correct;
#   - always emit an explicit TIMEOUT marker -- a container that lives but never serves is
#     otherwise indistinguishable from "still booting" to a log watcher;
#   - capture the TAIL of a traceback, never the head (the head is always boilerplate).
: "${IMG:?set IMG}"; : "${MODELS:?set MODELS}"; : "${LOGDIR:=$(pwd)}"

say(){ echo -e "\n########## $*" >> "$LOG"; }

serve(){ # name port model [extra docker args...] [--ARGS-- extra server args...]
  local name=$1 port=$2 model=$3; shift 3
  local denv=() sargs=()
  while [ $# -gt 0 ] && [ "$1" != "--ARGS--" ]; do denv+=("$1"); shift; done
  [ $# -gt 0 ] && shift; sargs=("$@")
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --gpus all --ipc=host --shm-size=8g \
    -v "$MODELS":/models:ro -p "$port":8000 "${denv[@]}" "$IMG" \
    --model "/models/$model" --served-model-name test \
    --tensor-parallel-size 2 --disable-custom-all-reduce --max-model-len 8192 \
    --gpu-memory-utilization 0.85 "${sargs[@]}" >/dev/null 2>&1
  for i in $(seq 1 95); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "READY ${i}0s" >> "$LOG"; return 0; }
    docker ps --filter "name=$name" --format '{{.Names}}' | grep -q "$name" || {
      echo "DIED at ${i}0s" >> "$LOG"
      docker logs "$name" 2>&1 | grep -aiE "Error|Exception|assert|raise " | tail -6 >> "$LOG"
      return 1; }
    sleep 10
  done
  echo "TIMEOUT (container alive but never served)" >> "$LOG"; return 1
}

fired(){ echo "  famp fired: $(docker logs "$1" 2>&1 | grep -ac 'FLASHAMPERE .* FIRED')" >> "$LOG"
         docker logs "$1" 2>&1 | grep -a 'FLASHAMPERE .* FIRED' | tail -2 >> "$LOG"; }
backends(){ docker logs "$1" 2>&1 | grep -aE "Using .*attention backend" | grep -avi all-reduce | sort -u >> "$LOG"; }
accept(){ docker logs "$1" 2>&1 | grep -a "acceptance length" | tail -1 >> "$LOG"; }
