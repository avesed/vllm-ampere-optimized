#!/usr/bin/env bash
# I-4b e2e runner: FA baseline + INT8QK, single-card 64k (or TP for 128k).
# Usage: TGT=65536 N=3 GPUMEM=0.60 GPUS=0 TP=1 BACKENDS="FA INT8QK" bash run_i4b.sh
set -u
IMG=ghcr.io/avesed/vllm-ampere-optimized:v0.23.0
D=/mnt/coder/workspaces/trevor/d2m
TGT=${TGT:-65536}
N=${N:-3}
GPUMEM=${GPUMEM:-0.60}
GPUS=${GPUS:-0}
TP=${TP:-1}
RUN_ZHCOT=${RUN_ZHCOT:-1}
BACKENDS=${BACKENDS:-"FA INT8QK"}
# CHUNKED prefill: set MBT (max_num_batched_tokens) < TGT to split the prompt into chunks.
# 0 (default) = single-step (the original I-4b behavior). e.g. MBT=8192 for 128k chunked.
MBT=${MBT:-0}

for B in $BACKENDS; do
  echo "############## BACKEND=$B TARGET=$TGT TP=$TP GPUS=$GPUS $(date +%T) ##############"
  docker rm -f i4be2e >/dev/null 2>&1
  # APPLY the flashinfer int8 patches (needed only for INT8QK, harmless for FA since f16 modules
  # are SFINAE-unaffected) THEN run the harness. rm flashinfer JIT cache so headers rebuild.
  docker run --rm --name i4be2e --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HOME=/out \
    -e TEST_BACKEND=$B -e TARGET_LEN=$TGT -e N_RUNS=$N -e GPU_MEM=$GPUMEM -e TP=$TP \
    -e RUN_ZHCOT=$RUN_ZHCOT -e MAX_BATCHED_TOKENS=$MBT \
    --shm-size=8g --ipc=host \
    -v "$D/models/Qwen3.5-9B-w4a8:/model" -v "$D:/out" \
    -v "$D/int8qk_backend.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/int8qk_backend.py:ro" \
    --entrypoint bash "$IMG" -lc "
      set -e
      if [ \"$B\" = INT8QK ]; then
        echo '--- applying flashinfer int8 patches ---'
        python3 /out/i1_apply.py >/tmp/i1.log 2>&1 || { echo I1_FAIL; tail -5 /tmp/i1.log; exit 1; }
        python3 /out/i4_apply.py >/tmp/i4a.log 2>&1 || { echo I4A_FAIL; tail -5 /tmp/i4a.log; exit 1; }
        python3 /out/i4_compute_qk.py >/tmp/i4c.log 2>&1 || { echo I4C_FAIL; tail -5 /tmp/i4c.log; exit 1; }
        rm -rf /out/.cache/flashinfer
        echo '--- patches applied ---'
      fi
      python3 /out/i4b_e2e.py
    " 2>&1 | grep -iE "RESULT|I4B_DONE|>>> |applied|FAIL|backend=|TTFT|fire|SPLIT|chunked|needle|zh_cot|error|traceback|out of memory|oom|raise |not supported|assert|JIT|Compil|nan|Unknown" | tail -70
  echo "---- end $B ----"
done
echo "ALL_I4B_DONE $(date +%T)"
