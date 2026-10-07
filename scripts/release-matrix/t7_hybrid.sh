#!/bin/bash
# T7 -- hybrid GDN (Qwen3.5-9B) on Ampere: GDN prefill must be Triton/FLA (FlashInfer GDN is
# SM90+), no torch fallback for the fla / causal-conv1d kernels, and a quality spot check.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t7.log; : > "$LOG"; . ./lib.sh

say "T7.1/T7.2 Qwen3.5-9B W4A16"
if serve t7-9b 8182 Qwen3.5-9B-w4a16; then
  docker logs t7-9b 2>&1 | grep -aE "GDN prefill kernel|falling back|fallback|causal_conv1d|fla " | sort -u | head -6 >> "$LOG"
  backends t7-9b
  gsm8k 8182 9b-w4a16 "${GSM8K_N:-40}"; canary 8182
fi
matrix_rm t7-9b
echo -e "\nT7_DONE" >> "$LOG"
