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

## C++ 部署 (NVIDIA Thor / TRT >= 8.5)

`deploy/cpp/`：完整 C++ 推理（图像+标定+can_bus → 轨迹），只用 TRT name-based
API（`setTensorAddress`/`enqueueV3`，TRT 10 上 bindings API 已删除，故兼容 Thor）。

```bash
# 构建（Thor 上 TensorRT/CUDA 在系统路径；自定义安装加 -DTENSORRT_ROOT=...）
cd deploy/cpp && mkdir -p build && cd build
cmake .. && make -j

# 第一步：raw 对齐验证（绕过 C++ 预处理，必须与 python 输出一致）
python - <<'PY'
import numpy as np, os
d = np.load('deploy/sparsedrive_v2_fp32_io.npz'); os.makedirs('deploy/raw', exist_ok=True)
for k in ["imgs", "projection_mat", "image_wh", "status_feature"]:
    d[k].astype('float32').tofile(f'deploy/raw/{k}.bin')
print("trajectory_ref:\n", d["trajectory_ref"][0])
PY
./sparsedrive_infer ../../sparsedrive_v2_fp32.engine --raw ../../raw
#   ↑ 输出应与上面 trajectory_ref 逐点一致（同一引擎、同一输入）

# 第二步：完整管线（按 calib_example.yaml 填你的图像路径与标定）
./sparsedrive_infer ../../sparsedrive_v2_fp32.engine --calib ../calib_example.yaml
```

C++ 预处理与 Python 的差异点（仅一处）：`cv::resize(INTER_CUBIC)` vs
`PIL.Image.resize`（插值实现细节不同，亚像素级差异）。raw 模式可逐位对齐；
完整管线模式如出现 argmax 漂移，用同一张图分别 dump 两边的 `imgs` 张量比对。
自采数据若有镜头畸变，喂入前先去畸变并替换为矫正后内参（模型为纯针孔投影）。

## PyTorch 推理 + 可视化 + 加载 OpenScene 自采数据

```bash
# A) 单帧推理 + 可视化（3 相机, 前视叠投影轨迹, BEV 轨迹+自车）
python deploy/infer.py --ckpt ckpt/sparsedrive_navsimv2_90p3.ckpt --version v2 \
    --cache-token exp/data_cache_mini/<log>/<token> --viz vis.png [--ground-z 0.0] [--cpu-daf]

# B) 直接加载“已转成 OpenScene 格式+pkl”的自采数据，批量推理+出图
python deploy/run_openscene.py --ckpt ckpt/sparsedrive_navsimv2_90p3.ckpt --version v2 \
    --data-root /path/to/your_dataset --split my_split --out-dir exp/openscene_vis \
    [--limit 20] [--no-route] [--ground-z 0.0] [--cpu-daf]
```
B 通过 navsim `SceneLoader` 直接读 `navsim_logs/<split>/*.pkl` + `sensor_blobs/<split>`，
每个 token 取 `AgentInput` 的当前帧相机/ego，喂入 `SparseDriveInference`，输出 `<token>.png`。
数据无 route 信息时加 `--no-route`（关闭 has_route 过滤）。
前视投影轨迹若浮在路面上方，调 `--ground-z` 为负值（约 -LiDAR 高度）。

## 已知限制

- batch 固定为 1（静态 shape，部署常态；需动态 batch 自行加 dynamic_axes）。
- v1 权重用 `--version v1`（自动切 6 指标头与 v1 加权公式）。
- 引擎含 ~26 万条词表常量与两层 decoder 的 topk/gather，TRT 构建耗时数分钟属正常。
- TF32 必须关（`--noTF32`），否则矩阵乘走 TF32 精度，argmax 可能在分数接近的候选间漂移。
