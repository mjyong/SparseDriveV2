# SparseDriveV2 → TensorRT (float32) 部署

## 原理

模型唯一不可导出的部分是自定义 CUDA 算子 `deformable_aggregation_ext`。
`daf_torch_patch.py` 用 `F.grid_sample` 给出**数学等价**实现（对照
`ops/src/deformable_aggregation_cuda.cu` 逐语义复刻：`align_corners=False` 坐标约定、
四角越界置零、`loc∈(0,1)` 相机门控、(cam,level,group) 加权求和），
替换后整个网络只剩 ONNX/TRT 原生支持的算子（Conv/MHA/topk/gather/grid_sample/argmax）。

注意：导出的是 **模型前向**（imgs → trajectory）。图像 resize/归一化等数据预处理
仍在 ONNX 之外（与 `sparsedrive_features.py:pipeline(test_mode=True)` 保持一致）。

## 依赖

- 训练环境（torch 2.0.1，ops 已编译——仅算子等价性测试需要）
- `pip install onnx onnxruntime`
- TensorRT **>= 8.5**（GridSample 层要求），FP32 严格模式用 `--noTF32`

## 步骤（按序执行，每步带数值校验）

```bash
# 1. 算子级等价性验证（GPU；CUDA 算子 vs grid_sample 实现，max|diff| 应 <1e-4）
python deploy/test_daf_equivalence.py

# 2. 导出 ONNX + PyTorch<->onnxruntime 校验（推荐用真实缓存样本）
python deploy/export_onnx.py \
    --ckpt ckpt/sparsedrive_navsimv2_90p3.ckpt --version v2 \
    --out deploy/sparsedrive_v2_fp32.onnx \
    --cache-sample exp/data_cache_mini/<log_name>/<token>

# 3. 构建 TRT 引擎（严格 fp32：--noTF32 关闭 Ampere 默认 TF32）
bash deploy/build_trt.sh deploy/sparsedrive_v2_fp32.onnx deploy/sparsedrive_v2_fp32.engine

# 4. TRT 引擎 vs PyTorch 数值校验
python deploy/verify_trt.py --engine deploy/sparsedrive_v2_fp32.engine \
                            --io deploy/sparsedrive_v2_fp32_io.npz
```

## 通过标准

| 级别 | 比较对象 | 阈值 |
|---|---|---|
| 算子级 | CUDA DAF vs grid_sample 实现 | max\|diff\| < 1e-4 |
| 图级 | PyTorch vs onnxruntime | max\|diff\| < 1e-3 |
| 引擎级 | PyTorch vs TensorRT | 轨迹逐点 < 1e-3 且**选中同一条词表轨迹** |

最后一条是硬标准：SparseDriveV2 的输出是"从离散词表 argmax 选一条"，
只要 argmax 不漂移，输出轨迹与 PyTorch **逐点完全一致**（它们是同一份常量词表里的同一行）。

## 已知限制

- batch 固定为 1（静态 shape，部署常态；需动态 batch 自行加 dynamic_axes）。
- v1 权重用 `--version v1`（自动切 6 指标头与 v1 加权公式）。
- 引擎含 ~26 万条词表常量与两层 decoder 的 topk/gather，TRT 构建耗时数分钟属正常。
- TF32 必须关（`--noTF32`），否则矩阵乘走 TF32 精度，argmax 可能在分数接近的候选间漂移。
