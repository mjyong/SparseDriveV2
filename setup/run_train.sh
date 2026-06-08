#!/usr/bin/env bash
# 训练入口薄封装。用法：
#   conda activate navsim && source setup/env.sh
#   bash setup/run_train.sh v1     # 或 v2
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?请先 source setup/env.sh}"
cd "$NAVSIM_DEVKIT_ROOT"

VER="${1:-v1}"
case "$VER" in
  v1) sh scripts/training/sparsedrive_navsimv1.sh ;;   # -> run_training.py, train_test_split=navtrain
  v2) sh scripts/training/sparsedrive_navsimv2.sh ;;
  *)  echo "用法: bash setup/run_train.sh [v1|v2]"; exit 1 ;;
esac
