#!/usr/bin/env bash
# Gate the publish: boot the image with a tiny model, assert /health, and assert the
# patched Marlin W4A8 scheme imports. Skips gracefully without a GPU — run it locally on the GPU
# box after building the image (scripts/build_image_source.sh). Usage: smoke_test.sh <image>
set -euo pipefail

IMG="${1:?usage: smoke_test.sh <image>}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::no GPU here — skipping runtime smoke test for $IMG"
  echo "(run this on the GPU box after building the image locally)"
  exit 0
fi

docker pull "$IMG"
cid=$(docker run -d --gpus all -p 8000:8000 "$IMG" \
  --model Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048 --enforce-eager)
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

ok=
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  echo "::error::server did not become healthy"; docker logs "$cid" | tail -60; exit 1
fi

# the whole value-add: the patched Marlin W4A8 scheme must import inside the image
docker exec "$cid" python -c \
  "import vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int as m; print('W4A8 scheme import OK')"

echo "smoke test passed for $IMG"
