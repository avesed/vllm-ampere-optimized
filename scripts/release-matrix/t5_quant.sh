#!/bin/bash
# T5 -- quant matrix: dense W4A16 / W4A8 (famp-marlin), MoE W4A16 / int8-act MoE, W8A8, and the
# FAMP_MARLIN=0 escape hatch. GSM8K numbers are only comparable to this block on another image.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t5.log; : > "$LOG"; . ./lib.sh
N=${GSM8K_N:-40}

marlin(){ # name -- which linear kernel and whether famp-marlin registered
  docker logs "$1" 2>&1 | grep -aE "famp_marlin: |Using .*LinearKernel|selected .*Marlin" | sort -u | head -4 >> "$LOG"
}

say "T5.1 27B W4A16"
if serve t5-w4a16 8185 Qwen3.6-27B-W4A16; then
  marlin t5-w4a16; gsm8k 8185 27b-w4a16 "$N"; canary 8185; fi
matrix_rm t5-w4a16

say "T5.2 27B W4A8 (--marlin-input-dtype int8)"
if serve t5-w4a8 8185 Qwen3.6-27B-W4A16 --ARGS-- --marlin-input-dtype int8; then
  marlin t5-w4a8; gsm8k 8185 27b-w4a8 "$N"; canary 8185; fi
matrix_rm t5-w4a8

say "T5.3 35B-A3B W4A16 (MoE)"
if serve t5-moe 8185 Qwen3.6-35B-A3B-W4A16; then
  docker logs t5-moe 2>&1 | grep -aE "Using configuration from .*E=256" | sort -u | head -2 >> "$LOG"
  gsm8k 8185 35b-w4a16 "$N"; canary 8185; fi
matrix_rm t5-moe

say "T5.4 35B-A3B int8-act MoE (--marlin-input-dtype int8)"
if serve t5-moe8 8185 Qwen3.6-35B-A3B-W4A16 --ARGS-- --marlin-input-dtype int8; then
  gsm8k 8185 35b-w4a8 "$N"; canary 8185; fi
matrix_rm t5-moe8

say "T5.5 27B W8A8"
if serve t5-w8a8 8185 Qwen3.6-27B-W8A8; then
  gsm8k 8185 27b-w8a8 "$N"; canary 8185; fi
matrix_rm t5-w8a8

say "T5.7 FAMP_MARLIN=0 escape hatch -> stock Marlin"
if serve t5-stock 8185 Qwen3.6-27B-W4A16 -e FAMP_MARLIN=0 --ARGS-- --marlin-input-dtype int8; then
  marlin t5-stock; canary 8185; fi
matrix_rm t5-stock
echo -e "\nT5_DONE" >> "$LOG"
