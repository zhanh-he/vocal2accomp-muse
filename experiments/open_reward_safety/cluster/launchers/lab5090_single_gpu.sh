#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  "${REPO_DIR}/experiments/open_reward_safety/cluster/distributed_smoke.py" \
  --payload-mb "${PAYLOAD_MB:-64}" \
  --iterations "${ITERATIONS:-8}"
