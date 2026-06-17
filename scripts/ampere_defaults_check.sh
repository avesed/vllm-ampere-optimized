#!/usr/bin/env bash
# Tier-A anti-regression: load a tiny hybrid (GatedDeltaNet) model in the SHIPPED image on an Ampere GPU
# and assert the runtime SELECTS the right Ampere paths — GDN prefill -> Triton/FLA (not a missing or
# Hopper-only fast path), attention -> FlashAttention. The kernel tests (ampere_kernel_ci.sh) call kernels
# directly; this catches a different regression class: an upstream bump silently routing Ampere to a slow
# or broken path. Skips gracefully without a GPU. Usage: ampere_defaults_check.sh <image>
set -euo pipefail

IMG="${1:?usage: ampere_defaults_check.sh <image>}"
MODEL="${2:-Qwen/Qwen3.5-0.8B-Base}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::no GPU on this runner — skipping Ampere defaults check for $IMG"
  exit 0
fi

docker pull "$IMG"
log="$(docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash "$IMG" -lc "
  python3 -c '
from vllm import LLM, SamplingParams
llm = LLM(model=\"$MODEL\", max_model_len=2048, enforce_eager=True,
          gpu_memory_utilization=0.6, limit_mm_per_prompt={\"image\": 0, \"video\": 0})
llm.generate([\"hello\"], SamplingParams(max_tokens=4))
print(\"DEFAULTS_SMOKE_OK\")
' 2>&1" || true)"

echo "$log" | tail -25

fail=0
echo "$log" | grep -q "Triton/FLA GDN prefill"       || { echo "::error::GDN prefill did NOT resolve to Triton/FLA on Ampere"; fail=1; }
echo "$log" | grep -q "FLASH_ATTN attention backend"  || { echo "::error::attention backend is NOT FlashAttention on Ampere"; fail=1; }
echo "$log" | grep -q "DEFAULTS_SMOKE_OK"             || { echo "::error::model failed to load/generate in the image"; fail=1; }
[ "$fail" = 0 ] || exit 1

echo "Ampere defaults check passed for $IMG (GDN prefill=Triton/FLA, attention=FlashAttention)"
