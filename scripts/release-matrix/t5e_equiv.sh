#!/bin/bash
# T5e -- famp-marlin must stay BIT-EXACT against stock MarlinLinearKernel (same weights, same
# inputs, torch.equal), run against the .so and Python baked into the image.
set -u
cd "$(dirname "$0")"; LOG=${LOGDIR:-$(pwd)}/t5e.log; : > "$LOG"; . ./lib.sh
say "T5e famp-marlin vs stock Marlin bit-exactness (in-image test_kernel_equiv)"
matrix_rm t5e-equiv
docker run --rm --name t5e-equiv --label "$MATRIX_LABEL" --gpus device=0 --entrypoint python3 \
  "$IMG" -m flashampere.marlin.test_kernel_equiv 2>&1 | grep -aE "EQUIV_OK|PASSED|Error|assert|famp != stock" | tail -10 >> "$LOG"
echo -e "\nT5E_DONE" >> "$LOG"
