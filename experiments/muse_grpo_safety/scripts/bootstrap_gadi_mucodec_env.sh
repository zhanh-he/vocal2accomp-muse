#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/g/data/wa66/hanyu/vocal2accomp-muse}"
ENV_DIR="${ENV_DIR:-${REPO_DIR}/envs/mucodec-py310-cu128}"
REQUIREMENTS="${REPO_DIR}/experiments/muse_grpo_safety/requirements-gadi-mucodec-py310.txt"

export PYTHONNOUSERSITE=1
module purge
module load python3/3.10.4
module load cuda/12.8.0

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  python3 -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install \
  pip==23.2.1 setuptools==69.5.1 wheel==0.43.0
"${ENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

export C_INCLUDE_PATH="/apps/python3/3.10.4/include/python3.10${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="/apps/python3/3.10.4/include/python3.10${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
"${ENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}"
"${ENV_DIR}/bin/python" -m pip check
"${ENV_DIR}/bin/python" -c \
  'import diffusers, fairseq, fastapi, librosa, nnAudio, numpy, torch, torchaudio, transformers, uvicorn; print(torch.__version__, torchaudio.__version__, numpy.__version__, diffusers.__version__)'
