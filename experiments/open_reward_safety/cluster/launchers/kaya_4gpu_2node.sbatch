#!/usr/bin/env bash
#SBATCH --job-name=muse-ddp-4g-2n-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=2
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
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${HOSTS[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE=2
export REPO_DIR PYTHON_BIN
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  "${REPO_DIR}/experiments/open_reward_safety/cluster/launchers/torchrun_node.sh"
