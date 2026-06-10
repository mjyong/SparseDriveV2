#!/usr/bin/env bash
# ONNX -> TensorRT 引擎（严格 float32）
# 要求: TensorRT >= 8.5 (GridSample 层自 8.5 起支持)
# 用法: bash deploy/build_trt.sh deploy/sparsedrive_fp32.onnx deploy/sparsedrive_fp32.engine
set -euo pipefail
ONNX=${1:-deploy/sparsedrive_fp32.onnx}
ENGINE=${2:-deploy/sparsedrive_fp32.engine}

# --noTF32: 关闭 Ampere 默认的 TF32，保证严格 fp32（这是"保持 float32"的关键开关）
trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --noTF32 \
    --memPoolSize=workspace:4096 \
    --verbose 2>&1 | tail -20

echo "[ok] engine -> $ENGINE"
