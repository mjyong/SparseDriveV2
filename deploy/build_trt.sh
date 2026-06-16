#!/usr/bin/env bash
# ONNX -> TensorRT 引擎（严格 float32）
# 要求: TensorRT >= 8.5 (GridSample 层自 8.5 起支持)
# 用法: bash deploy/build_trt.sh deploy/sparsedrive_v2_fp32.onnx deploy/sparsedrive_v2_fp32.engine
set -euo pipefail
ONNX=${1:-deploy/sparsedrive_fp32.onnx}
ENGINE=${2:-deploy/sparsedrive_fp32.engine}

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
echo "[info] onnx   = $ONNX"
echo "[info] engine = $ENGINE"

# --noTF32: 关闭 Ampere 默认的 TF32，保证严格 fp32（这是"保持 float32"的关键开关）
trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --noTF32 \
    --memPoolSize=workspace:4096 \
    --verbose 2>&1 | tail -20

echo "[ok] engine -> $ENGINE"

