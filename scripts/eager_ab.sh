#!/bin/bash
# Eager A/B: with cudagraph OFF the verify path always fires (no FULL-capture fallback). Proves
# whether routing verify->fwd_kvcache actually speeds up e2e MTP decode when it fires.
set -u
cp /home/coder/run_api_bench.sh /home/coder/run_api_bench.sh.eagerbak
sed -i 's/--trust-remote-code "\$@"/--trust-remote-code --enforce-eager "$@"/' /home/coder/run_api_bench.sh
cd /home/coder
for V in 1 0; do
  echo "######## eager VLLM_FA2_KVCACHE_VERIFY=$V ########"
  : > apibench.log
  MODEL_OVERRIDE=/home/coder/models/Qwen3.5-9B-w4a16 TP_OVERRIDE=1 CVD_OVERRIDE=0 \
    VLLM_FA2_KVCACHE_VERIFY=$V bash run_api_bench.sh MTP > /dev/null 2>&1
  grep RESULT apibench.log
  echo "fired-once-flag: $(grep -c VERIFY-PATH-FIRED serve.log)"
done
cp /home/coder/run_api_bench.sh.eagerbak /home/coder/run_api_bench.sh
echo "[eager A/B done]"
