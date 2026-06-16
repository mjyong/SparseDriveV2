#!/usr/bin/env bash
# ONNX -> TensorRT 引擎（严格 float32）
# 要求: TensorRT >= 8.5 (GridSample 层自 8.5 起支持)
# 用法: bash deploy/build_trt.sh <onnx> <engine>
# 可调环境变量:
#   WORKSPACE=16384      工作区 MiB (Myelin 大节点编译吃显存，Time:inf 常因不够)
#   OPTLEVEL=2           builderOptimizationLevel 0-5，降级可避免激进 Myelin 大融合
#   NOTF32=1             1=严格 fp32(--noTF32); 0=允许 TF32(tactic 更多，先验证能否 build)
set -euo pipefail
ONNX=${1:-deploy/sparsedrive_fp32.onnx}
ENGINE=${2:-deploy/sparsedrive_fp32.engine}
WORKSPACE=${WORKSPACE:-16384}
OPTLEVEL=${OPTLEVEL:-2}
NOTF32=${NOTF32:-1}

# 先检查 ONNX 是否存在（trtexec 找不到文件只会报 INVALID_VALUE，很迷惑）
if [ ! -f "$ONNX" ]; then
    echo "[error] ONNX 不存在: $ONNX"
    echo "        - 先跑 export_onnx.py 生成它，且 --out 名字要和这里一致"
    echo "        - 确认在仓库根目录运行（deploy/ 是相对路径）"
    echo "        - 当前目录已有的 onnx:"
    ls -la deploy/*.onnx 2>/dev/null || echo "          (deploy/ 下没有任何 .onnx)"
    exit 1
fi
# 转成绝对路径，避免 cwd 不对导致找不到
ONNX=$(readlink -f "$ONNX")
ENGINE=$(readlink -f "$(dirname "$ENGINE")")/$(basename "$ENGINE")

TF32_FLAG=""
[ "$NOTF32" = "1" ] && TF32_FLAG="--noTF32"
echo "[info] onnx=$ONNX"
echo "[info] engine=$ENGINE  workspace=${WORKSPACE}MiB  optLevel=$OPTLEVEL  noTF32=$NOTF32"

# --noTF32: 严格 fp32(关 Ampere TF32)。--builderOptimizationLevel 降级避免 Myelin 巨融合。
trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    $TF32_FLAG \
    --builderOptimizationLevel="$OPTLEVEL" \
    --memPoolSize=workspace:"$WORKSPACE" \
    --verbose 2>&1 | tail -30

echo "[ok] engine -> $ENGINE"


