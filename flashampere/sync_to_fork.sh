#!/bin/bash
# Source-of-truth -> build deploy.
#
# This directory (project-root /flashampere) is the SINGLE source of truth for the flashampere
# Ampere attention backend (the Backend.CUSTOM plugin) + its vendored kernels. All development
# happens HERE. vLLM can only load it from inside its package tree, so this script deploys a copy
# into the fork's backend path, which the image build / runtime then picks up.
#
# Usage:   ./sync_to_fork.sh            (uses the default fork path below)
#          FORK_BACKEND=/path/... ./sync_to_fork.sh
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_BACKEND="${FORK_BACKEND:-/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/attention/backends/flashampere}"

if [ ! -d "$(dirname "$FORK_BACKEND")" ]; then
  echo "[sync] ERROR: fork backends dir $(dirname "$FORK_BACKEND") not found; set FORK_BACKEND=" >&2
  exit 1
fi
mkdir -p "$FORK_BACKEND"
rsync -a --delete \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='sync_to_fork.sh' --exclude='README.md' --exclude='vendor/' \
  "$SELF"/ "$FORK_BACKEND"/
echo "[sync] flashampere (source-of-truth) -> $FORK_BACKEND"
