#!/usr/bin/env bash
# SparseDriveV2 数据 / 地图 / 权重 / anchor / 缓存 准备
#
# 前置： conda activate navsim && source setup/env.sh
# 说明： 本脚本分步骤，建议第一次按需注释/逐段运行，数据集很大(数百 GB)。
set -euo pipefail

: "${OPENSCENE_DATA_ROOT:?请先 source setup/env.sh}"
: "${NUPLAN_MAPS_ROOT:?请先 source setup/env.sh}"
: "${NAVSIM_EXP_ROOT:?请先 source setup/env.sh}"
cd "$NAVSIM_DEVKIT_ROOT"

# ---------------------------------------------------------------------------
echo "### [A] nuPlan 地图  ->  \$NUPLAN_MAPS_ROOT/maps"
# download/download_maps.sh 会下载 nuplan-maps-v1.1 并解压成 maps/
mkdir -p "$NUPLAN_MAPS_ROOT"
if [ ! -d "$NUPLAN_MAPS_ROOT/maps" ]; then
    ( cd "$NUPLAN_MAPS_ROOT" && bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh" )
else
    echo "  已存在 $NUPLAN_MAPS_ROOT/maps ，跳过"
fi

# ---------------------------------------------------------------------------
echo "### [B] 数据集 (OpenScene/NAVSIM)  ->  \$OPENSCENE_DATA_ROOT"
# 目标布局(config 要求)：
#   $OPENSCENE_DATA_ROOT/navsim_logs/{trainval,test}
#   $OPENSCENE_DATA_ROOT/sensor_blobs/{trainval,test}
# 其中 navtrain->data_split=trainval, navtest->data_split=test
#
# 下面命令会下数百 GB，按需运行。download 脚本产出名是 *_navsim_logs / *_sensor_blobs，
# 需按下方 mv 重排成 config 期望布局。
mkdir -p "$OPENSCENE_DATA_ROOT" "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
cat <<EOF
  手动执行（首次）：
  # --- 训练集 navtrain (=trainval) ---
  cd "$OPENSCENE_DATA_ROOT"
  bash "$NAVSIM_DEVKIT_ROOT/download/download_navtrain_hf.sh"
  mv trainval_navsim_logs        navsim_logs/trainval
  mv trainval_sensor_blobs/trainval sensor_blobs/trainval

  # --- 测试集 navtest (=test) ---
  bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"
  mv test_navsim_logs   navsim_logs/test
  mv test_sensor_blobs  sensor_blobs/test

  # (可选，快速跑通用) mini 集：
  bash "$NAVSIM_DEVKIT_ROOT/download/download_mini.sh"   # -> navsim_logs/mini, sensor_blobs/mini
EOF

# ---------------------------------------------------------------------------
echo "### [C] 预训练权重与 anchor  ->  \$NAVSIM_DEVKIT_ROOT/ckpt"
# config 默认相对路径： bkb_path=ckpt/resnet34.bin
#   path_anchor=ckpt/kmeans/path_1024.npy
#   velocity_anchor=ckpt/kmeans/velocity_256.npy
#   trajectory_anchor=ckpt/kmeans/trajectory_1024_256.npz
mkdir -p ckpt/kmeans
# ResNet-34 backbone
if [ ! -f ckpt/resnet34.bin ]; then
    wget -O ckpt/resnet34.bin \
      "https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/pytorch_model.bin"
fi
cat <<EOF
  anchor 与 SparseDriveV2 权重（二选一）：
  (1) 直接下载作者提供的 anchor / 权重到 ckpt/ 和 ckpt/kmeans/：
      https://huggingface.co/wenchaosun/SparseDriveV2
      需要：ckpt/kmeans/path_1024.npy, velocity_256.npy, trajectory_1024_256.npz
      推理还需：ckpt/sparsedrive_navsimv1.ckpt （或 v2 权重）
  (2) 或在已完成 [D] 数据缓存后，自行聚类生成 anchor：
      python scripts/cluster/cluster_anchor.py   # 读 exp/data_cache_navtrain，写 ckpt/kmeans/
EOF

# ---------------------------------------------------------------------------
echo "### [D] 数据缓存 + 指标缓存（训练/评测前必须）"
cat <<EOF
  数据缓存（特征/target -> \$NAVSIM_EXP_ROOT 即 exp/）：
    sh scripts/cache/run_dataset_caching_navtrain.sh
    sh scripts/cache/run_dataset_caching_navtest.sh
  指标缓存（PDM 闭环监督/评测用）：
    # navsimv1
    sh scripts/cache/run_metric_caching_navtrain_v1.sh
    sh scripts/cache/run_metric_caching_navtest_v1.sh
    # navsimv2
    sh scripts/cache/run_metric_caching_navtrain_v2.sh
    sh scripts/cache/run_metric_caching_navtest_v2.sh
EOF

echo "准备脚本结束（数据/权重下载请按上面提示手动执行）。"
