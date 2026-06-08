#!/usr/bin/env bash
# 推理/评测入口薄封装（跑 PDM 打分，输出 csv 到 $NAVSIM_EXP_ROOT）。用法：
#   conda activate navsim && source setup/env.sh
#   bash setup/run_infer.sh v1     # 或 v2 / navhard
#
# 注意：评测脚本里默认 CHECKPOINT=ckpt/sparsedrive_navsimv1.ckpt（v2 同理），
#       请先把权重放到 ckpt/ ，或编辑对应 scripts/evaluation/*.sh 改 CHECKPOINT。
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?请先 source setup/env.sh}"
cd "$NAVSIM_DEVKIT_ROOT"

VER="${1:-v1}"
case "$VER" in
  v1)      sh scripts/evaluation/run_pdm_score_navtest_v1.sh ;;  # -> run_pdm_score_navtest_v1_fast.py
  v2)      sh scripts/evaluation/run_pdm_score_navtest_v2.sh ;;
  navhard) sh scripts/evaluation/run_pdm_score_navhard.sh ;;
  *)       echo "用法: bash setup/run_infer.sh [v1|v2|navhard]"; exit 1 ;;
esac
