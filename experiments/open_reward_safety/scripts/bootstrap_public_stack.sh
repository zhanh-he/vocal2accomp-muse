#!/usr/bin/env bash
set -euo pipefail

: "${STUDY_ROOT:?Set STUDY_ROOT to an approved study directory}"
: "${ALLOW_NETWORK_SETUP:?Set ALLOW_NETWORK_SETUP=1 after approving downloads}"
if [[ "${ALLOW_NETWORK_SETUP}" != 1 ]]; then
  printf 'Refusing network setup: ALLOW_NETWORK_SETUP must equal 1\n' >&2
  exit 2
fi

mkdir -p "${STUDY_ROOT}/src" "${STUDY_ROOT}/receipts"

clone_once() {
  local url=$1
  local target=$2
  if [[ ! -d "${target}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${target}"
  fi
  git -C "${target}" rev-parse HEAD
}

{
  printf 'Muse '
  clone_once https://github.com/yuhui1038/Muse "${STUDY_ROOT}/src/Muse"
  printf 'MuseCritic '
  clone_once https://github.com/WuqnEl/MuseCritic "${STUDY_ROOT}/src/MuseCritic"
  printf 'SongEval '
  clone_once https://github.com/ASLP-lab/SongEval "${STUDY_ROOT}/src/SongEval"
  printf 'CMI-RewardBench '
  clone_once https://github.com/Haiwen-Xia/CMI-RewardBench "${STUDY_ROOT}/src/CMI-RewardBench"
} >"${STUDY_ROOT}/receipts/source_commits.txt"
