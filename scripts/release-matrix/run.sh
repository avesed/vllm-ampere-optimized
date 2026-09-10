#!/bin/bash
# Chain release-matrix blocks and write one spine log. Usage: run.sh t6_spec t2_batch t9_soak
set -u
cd "$(dirname "$0")"
export IMG="${IMG:?set IMG}" MODELS="${MODELS:?set MODELS}" LOGDIR="${LOGDIR:-$(pwd)}"
S="$LOGDIR/spine.log"; : > "$S"
for blk in "$@"; do
  echo "BLOCK_START ${blk%%_*} $(date +%H:%M:%S)" >> "$S"
  timeout 14400 "./$blk.sh" >/dev/null 2>&1
  echo "BLOCK_END ${blk%%_*} rc=$? $(date +%H:%M:%S)" >> "$S"
done
echo MATRIX_DONE >> "$S"
