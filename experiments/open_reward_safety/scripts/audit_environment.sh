#!/usr/bin/env bash
set -euo pipefail

printf 'host=%s\n' "$(hostname)"
printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'python=%s\n' "$(command -v python || true)"
printf 'ffmpeg=%s\n' "$(command -v ffmpeg || true)"
printf 'git=%s\n' "$(command -v git || true)"
printf 'slurm_sbatch=%s\n' "$(command -v sbatch || true)"
df -h "${PWD}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
else
  printf 'gpu=not_visible\n'
fi
