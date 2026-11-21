#!/bin/bash
# T5 -- quant matrix: dense W4A16 / W4A8 (Humming int8 activations), MoE W4A16 / W4A8, W8A8. GSM8K
# numbers are only comparable to this block on another image.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t5.log; : > "$LOG"; . ./lib.sh
N=${GSM8K_N:-40}

kern(){ # name -- which linear / MoE kernels were selected
  docker logs "$1" 2>&1 | grep -aE "Using [A-Za-z0-9]*(Kernel|Method|Experts) for|[Hh]umming" | sed "s/^.*\] //" | sort -u | head -4 >> "$LOG"
}

say "T5.1 27B W4A16"
if serve t5-w4a16 8185 Qwen3.6-27B-W4A16; then
  kern t5-w4a16; gsm8k 8185 27b-w4a16 "$N"; canary 8185; fi
matrix_rm t5-w4a16

say "T5.2 27B W4A8 (Humming, int8 per-token activations)"
if serve t5-w4a8 8185 Qwen3.6-27B-W4A16 "${HUM_ENV[@]}" --ARGS-- "${HUM_ARGS[@]}"; then
  kern t5-w4a8; gsm8k 8185 27b-w4a8 "$N"; canary 8185; fi
matrix_rm t5-w4a8

say "T5.3 35B-A3B W4A16 (MoE)"
if serve t5-moe 8185 Qwen3.6-35B-A3B-W4A16; then
  docker logs t5-moe 2>&1 | grep -aE "Using configuration from .*E=256" | sort -u | head -2 >> "$LOG"
  gsm8k 8185 35b-w4a16 "$N"; canary 8185; fi
matrix_rm t5-moe

say "T5.4 35B-A3B W4A8 MoE (Humming; crashes at load without the #68194 fix)"
if serve t5-moe8 8185 Qwen3.6-35B-A3B-W4A16 "${HUM_ENV[@]}" --ARGS-- "${HUM_ARGS[@]}"; then
  kern t5-moe8; gsm8k 8185 35b-w4a8 "$N"; canary 8185; fi
matrix_rm t5-moe8

say "T5.5 27B W8A8"
# W8A8 weights leave ~104 Mamba state blocks at 0.85 util; the default max_num_seqs=256 cannot capture.
if serve t5-w8a8 8185 Qwen3.6-27B-W8A8 --ARGS-- --max-num-seqs 64; then
  gsm8k 8185 27b-w8a8 "$N"; canary 8185; fi
matrix_rm t5-w8a8

echo -e "\nT5_DONE" >> "$LOG"
