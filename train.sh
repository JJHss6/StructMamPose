#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python train.py \
  --config pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint checkpoint/pose3d/PoseMamba_S_original_online_equivalent "$@"
