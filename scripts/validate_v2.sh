#!/bin/bash
# Patch A v2 = drop the cudagraph guard (fwd_kvcache proven capturable). Make it permanent on the
# sandbox, re-run 9B MTP, and capture the REAL accept-len from serve.log (the correctness canary:
# if the fwd_kvcache verify logits match varlen, accept-len must be unchanged ~2.0-2.5).
set -u
FA=/home/coder/vllm-build/vllm/v1/attention/backends/flash_attn.py
if grep -q "and not torch.cuda.is_current_stream_capturing()" "$FA"; then
  cp "$FA" "$FA.guardbak2"
  sed -i '/and not torch.cuda.is_current_stream_capturing()/d' "$FA"
fi
echo "is_current_stream_capturing refs now: $(grep -c is_current_stream_capturing "$FA")"
python3 -m py_compile "$FA" && echo "compile OK"
cd /home/coder
: > apibench.log
MODEL_OVERRIDE=/home/coder/models/Qwen3.5-9B-w4a16 TP_OVERRIDE=1 CVD_OVERRIDE=0 \
  VLLM_FA2_KVCACHE_VERIFY=1 bash run_api_bench.sh MTP > /dev/null 2>&1
grep RESULT apibench.log
echo "=== accept-len (serve.log; runs 4k/16k/32k) ==="
grep -aE "Mean acceptance length" serve.log | grep -oE "acceptance length: [0-9.]+" | tail -6
echo "[validate v2 done; guard stays removed]"
