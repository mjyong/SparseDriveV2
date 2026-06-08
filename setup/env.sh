# SparseDriveV2 环境变量
# 用法： source setup/env.sh   （每开一个新 shell 都要 source）
#
# 下面 4 个变量被 config / scripts 直接引用：
#   NAVSIM_DEVKIT_ROOT  -> 代码仓库根目录（scripts/*.sh 用它定位 python 入口）
#   OPENSCENE_DATA_ROOT -> 数据集根（navsim_logs/ 与 sensor_blobs/ 在它下面）
#   NAVSIM_EXP_ROOT     -> 实验/缓存输出根（data_cache、metric_cache、训练日志、评测 csv）
#   NUPLAN_MAPS_ROOT    -> 地图根（其下需有 maps/ ，map_version 固定 nuplan-maps-v1.0）
#
# ↓↓↓ 按你的机器改这几行（DEVKIT 一般不用改）↓↓↓
export NAVSIM_DEVKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/sparsedrive_data/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$HOME/sparsedrive_data/maps}"

# 重要：训练/评测脚本(scripts/training, scripts/evaluation)用相对路径 exp/ 和 ckpt/，
# 而缓存脚本(scripts/cache)用 $NAVSIM_EXP_ROOT。这里把 NAVSIM_EXP_ROOT 指到仓库内 exp/，
# 使两者落到同一目录，避免缓存找不到。如需放到别处，请把 exp/ 做成指向该处的软链接。
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"

# nuplan 地图版本（caching.py / dataclasses.py 里硬编码为 nuplan-maps-v1.0）
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# 让 python 能 import navsim
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

mkdir -p "$NAVSIM_EXP_ROOT"

echo "[env] NAVSIM_DEVKIT_ROOT = $NAVSIM_DEVKIT_ROOT"
echo "[env] OPENSCENE_DATA_ROOT= $OPENSCENE_DATA_ROOT"
echo "[env] NAVSIM_EXP_ROOT    = $NAVSIM_EXP_ROOT"
echo "[env] NUPLAN_MAPS_ROOT   = $NUPLAN_MAPS_ROOT"
