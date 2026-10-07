#!/bin/bash
# Chain release-matrix blocks and write one spine log. Usage: run.sh t6_spec t2_batch t9_soak
set -u
cd "$(dirname "$0")"
export IMG="${IMG:?set IMG}" MODELS="${MODELS:?set MODELS}" LOGDIR="${LOGDIR:-$(pwd)}"
S="$LOGDIR/spine.log"; : > "$S"
# A checkpoint with lost-write holes makes every number below meaningless (found 2026-10-07: three
# staged checkpoints carried 16-36 MiB zeroed extents). Refuse to run unless the holey checkpoint
# is explicitly allowed (CKPT_HOLES_OK="name1 name2", e.g. to compare against a published artifact).
python3 ckpt_holes.py "$MODELS" > "$LOGDIR/ckpt_holes.log" 2>&1
holes=$(grep "zero 4MiB blocks" "$LOGDIR/ckpt_holes.log" | cut -d/ -f1 | sort -u)
for ok in ${CKPT_HOLES_OK:-}; do holes=$(echo "$holes" | grep -vx "$ok"); done
if [ -n "$holes" ]; then
  echo "CKPT_HOLES in: $holes (see ckpt_holes.log) -- matrix not run" >> "$S"; exit 1
fi
for blk in "$@"; do
  echo "BLOCK_START ${blk%%_*} $(date +%H:%M:%S)" >> "$S"
  timeout 14400 "./$blk.sh" >/dev/null 2>&1
  echo "BLOCK_END ${blk%%_*} rc=$? $(date +%H:%M:%S)" >> "$S"
done
echo MATRIX_DONE >> "$S"
