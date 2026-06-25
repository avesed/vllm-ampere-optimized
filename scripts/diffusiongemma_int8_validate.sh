#!/usr/bin/env bash
# Run the DiffusionGemma int8 validation module (benchmarks/diffusiongemma_int8_validate.py) inside
# the from-source image (must contain BOTH patch 0005/0006 AND the cherry-picked PR #45163 python).
# Forces the fork's int8 Marlin path (VLLM_MARLIN_INPUT_DTYPE=int8) and asserts G1-G4.
# SINGLE GPU only — DiffusionGemma crashes on TP>1/PP>1 (vLLM #45719).
# Skips gracefully without a GPU. Usage: diffusiongemma_int8_validate.sh <image> <ckpt_dir> [gsm8k_n]
set -euo pipefail

IMG="${1:?usage: diffusiongemma_int8_validate.sh <image> <ckpt_dir> [gsm8k_n]}"
CKPT="${2:?usage: diffusiongemma_int8_validate.sh <image> <ckpt_dir> [gsm8k_n]}"
GSM8K_N="${3:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::no GPU here — skipping DiffusionGemma int8 validation for $IMG"
  echo "(run on the GPU box after building the cherry-picked image)"
  exit 0
fi
[ -d "$CKPT" ] || { echo "::error::ckpt dir not found: $CKPT"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
PY="$HERE/benchmarks/diffusiongemma_int8_validate.py"
DATA="$HERE/benchmarks/gsm8k.jsonl"
[ -f "$PY" ] || { echo "::error::missing $PY"; exit 1; }

# --flag-via-env: 0006's --marlin-input-dtype is a server arg; for the in-process LLM() probe we set
# the env directly (the kernel reads VLLM_MARLIN_INPUT_DTYPE). device=0 only (single card, #45719).
exec docker run --rm --gpus '"device=0"' --shm-size=8g \
  -e VLLM_MARLIN_INPUT_DTYPE=int8 \
  -v "$CKPT":"$CKPT":ro \
  -v "$PY":/work/validate.py:ro \
  -v "$DATA":/work/gsm8k.jsonl:ro \
  --entrypoint python "$IMG" \
  /work/validate.py --model "$CKPT" --data /work/gsm8k.jsonl \
  --max-model-len 8192 --gpu-mem-util 0.85 --max-num-seqs 4 --gsm8k "$GSM8K_N"
