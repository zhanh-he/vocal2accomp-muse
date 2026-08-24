#!/usr/bin/env bash
set -euo pipefail

: "${REPO_DIR:?REPO_DIR must point to vocal2accomp-muse}"
: "${PYTHON_BIN:?PYTHON_BIN must point to a CUDA-enabled Python}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${NNODES:?NNODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"

NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_DIR}/experiments/open_reward_safety/cluster/distributed_smoke.py" \
  --payload-mb "${PAYLOAD_MB:-64}" \
  --iterations "${ITERATIONS:-8}"
