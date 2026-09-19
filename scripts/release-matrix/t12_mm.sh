#!/bin/bash
# T12 -- multimodal. Images (and one video) through every surface the fork changes: Marlin W4A16,
# the int8-act W4A8 path (dense and MoE), the famp prefill legs, all three spec-decode methods, and
# the Gemma4 per-head backend split, where the hd256 sliding layers carry the bidirectional image
# prefix. Vision towers stay bf16 in every checkpoint, so what is under test is the language model
# consuming image embeddings. Probe = mm_probe.py (stdlib, OpenAI API); every answer is a colour,
# shape or count, so a model that is not actually seeing the pixels cannot pass by luck.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t12.log; : > "$LOG"; . ./lib.sh
VID=${LOGDIR:-$(pwd)}/t12_video.b64
P=8190

probe(){ python3 mm_probe.py "$P" "$@" >> "$LOG" 2>&1; }
dead(){ echo "  engine faults: $(docker logs "$1" 2>&1 | grep -acE 'EngineDeadError|CUDA error|illegal memory access')" >> "$LOG"; }
kern(){ docker logs "$1" 2>&1 | grep -aoE "famp_marlin: [^.]*|Using [A-Za-z]*Kernel for [A-Za-z0-9]*" | sort -u | head -4 | sed 's/^/  /' >> "$LOG"; }

# 16 frames @ 4 fps, a yellow circle drifting across a blue field. Built with the image's own cv2 so
# the clip is decodable by the same backend that will read it.
docker run --rm --label "$MATRIX_LABEL" --entrypoint python3 "$IMG" -c '
import base64, cv2, numpy as np
w = cv2.VideoWriter("/tmp/v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 4, (320, 240))
for i in range(16):
    f = np.full((240, 320, 3), (220, 60, 30), np.uint8)
    cv2.circle(f, (70 + i * 12, 120), 50, (20, 210, 240), -1)
    w.write(f)
w.release()
print(base64.b64encode(open("/tmp/v.mp4", "rb").read()).decode())' > "$VID" 2>/dev/null
say "T12.0 video clip: $([ -s "$VID" ] && echo "$(wc -c < "$VID") base64 bytes" || echo "NOT generated -- the video case will report SKIPPED")"

say "T12.1 27B W4A16, stock attention -- battery + video + corrupt input + W=8 mixed traffic"
if serve t12-base $P Qwen3.6-27B-W4A16; then
  probe battery --video "$VID"; probe concurrency 8 48; backends t12-base; kern t12-base; dead t12-base
fi
matrix_rm t12-base

say "T12.2 27B W4A16 served as W4A8 (--marlin-input-dtype int8)"
if serve t12-int8 $P Qwen3.6-27B-W4A16 --ARGS-- --marlin-input-dtype int8; then
  probe battery; probe concurrency 8 48; kern t12-int8; dead t12-int8
fi
matrix_rm t12-int8

say "T12.3 27B W4A16 + famp (bf16-served -> bf16cvt prefill leg on the hd256 layers)"
if serve t12-famp $P Qwen3.6-27B-W4A16 -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0; then
  probe battery --video "$VID"; probe concurrency 8 48; fired t12-famp; backends t12-famp; dead t12-famp
fi
matrix_rm t12-famp

spec_mm(){ # id name method draft|- K
  local id=$1 name=$2 method=$3 draft=$4 K=$5 cfg
  cfg="{\"method\":\"$method\",\"num_speculative_tokens\":$K"
  [ "$draft" != "-" ] && cfg+=",\"model\":\"/models/$draft\""
  cfg+="}"
  say "$id spec $method K=$K + images (verification is lossless, so answers must match T12.1)"
  if serve "$name" $P Qwen3.6-27B-W4A16 \
       -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_FLASHAMPERE_XQA_VERIFY=1 \
       --ARGS-- --max-num-seqs 128 --speculative-config "$cfg"; then
    probe battery --video "$VID"; probe concurrency 8 48; accept "$name"; dead "$name"
  fi
  matrix_rm "$name"
}
spec_mm T12.4 t12-mtp mtp - 2
spec_mm T12.5 t12-dflash dflash Qwen3.6-27B-DFlash 7
spec_mm T12.6 t12-dspark dspark dspark_27b_published 7

say "T12.7 35B-A3B MoE W4A16 served as W4A8 (int8-act MoE on image embeddings)"
if serve t12-moe $P Qwen3.6-35B-A3B-W4A16 --ARGS-- --marlin-input-dtype int8; then
  probe battery --video "$VID"; probe concurrency 8 48; kern t12-moe; dead t12-moe
fi
matrix_rm t12-moe

say "T12.8 gemma-4-12B + famp profile -- bidirectional image prefix on the hd256 sliding layers"
if serve t12-gemma $P gemma-4-12B-it-bf16 -e VLLM_FLASHAMPERE=1 -e VLLM_USE_V2_MODEL_RUNNER=0; then
  probe battery; probe concurrency 8 48; backends t12-gemma; fired t12-gemma; dead t12-gemma
fi
matrix_rm t12-gemma
echo -e "\nT12_DONE" >> "$LOG"
