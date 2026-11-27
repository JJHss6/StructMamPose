#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
variant="${1:-}"
case "$variant" in
  S|B) shift ;;
  *) echo 'Usage: bash train.sh S|B [train.py arguments]' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec 9>".training_${variant}.lock"
flock -n 9 || { echo "${variant} is already running in this directory." >&2; exit 1; }
python_bin="${POSEMAMBA_PYTHON:-python}"
exec "$python_bin" -u train.py \
  --config "pose3d/PoseMamba_train_h36m_${variant}.yaml" \
  --checkpoint "checkpoint/pose3d/PoseMamba_${variant}_seed0" \
  --seed 0 "$@"
