#!/bin/bash
# fwd_kvcache is cudagraph-capturable (proven). Drop the is_current_stream_capturing guard so the
# verify path is captured INTO the FULL cudagraph, then run the default (cudagraph) A/B to see the
# +12% land in the real serve. FIRED>=1 means it was captured (the one-shot print fires at capture).
set -u
FA=/home/coder/vllm-build/vllm/v1/attention/backends/flash_attn.py
cp "$FA" "$FA.guardbak"
sed -i '/and not torch.cuda.is_current_stream_capturing()/d' "$FA"
echo "remaining is_current_stream_capturing refs: $(grep -c is_current_stream_capturing "$FA")"
python3 -m py_compile "$FA" && echo "compile OK" || { echo "COMPILE FAIL"; cp "$FA.guardbak" "$FA"; exit 1; }
cd /home/coder
for V in 1 0; do
  echo "######## cudagraph VLLM_FA2_KVCACHE_VERIFY=$V ########"
  : > apibench.log
  MODEL_OVERRIDE=/home/coder/models/Qwen3.5-9B-w4a16 TP_OVERRIDE=1 CVD_OVERRIDE=0 \
    VLLM_FA2_KVCACHE_VERIFY=$V bash run_api_bench.sh MTP > /dev/null 2>&1
  grep RESULT apibench.log || { echo "[serve failed - tail]"; tail -15 serve.log; }
  echo "fired(captured): $(grep -c VERIFY-PATH-FIRED serve.log)"
done
cp "$FA.guardbak" "$FA"   # restore guarded version (decide after seeing results)
echo "[removeguard A/B done; guard restored]"
