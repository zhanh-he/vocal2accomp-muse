#!/usr/bin/env bash
#SBATCH --job-name=muse-ddp-2g-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-32gb:2
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/cluster/%x-%j.out
#SBATCH --error=artifacts/cluster/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "${REPO_DIR}/artifacts/cluster"

module load cuda/12.4.1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --ntasks=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "${REPO_DIR}/experiments/open_reward_safety/cluster/distributed_smoke.py" \
  --payload-mb "${PAYLOAD_MB:-64}" \
  --iterations "${ITERATIONS:-8}"
