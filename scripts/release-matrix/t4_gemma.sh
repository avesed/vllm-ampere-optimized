#!/bin/bash
# T4 -- Gemma4 (hd512 full-attn + hd256 sliding). famp must decline every layer (no FA2 'head
# dimension at most 256' crash), TRITON takes them, and quality holds.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t4.log; : > "$LOG"; . ./lib.sh

say "T4.1 gemma-4-12B-it bf16, famp profile on"
if serve t4-gemma 8181 gemma-4-12B-it-bf16 -e VLLM_FLASHAMPERE=1; then
  backends t4-gemma; fired t4-gemma
  echo "  hd>256 crashes: $(docker logs t4-gemma 2>&1 | grep -ac 'head dimension at most 256')" >> "$LOG"
  gsm8k 8181 gemma4-12b "${GSM8K_N:-40}"; canary 8181
fi
matrix_rm t4-gemma
echo -e "\nT4_DONE" >> "$LOG"
