#!/usr/bin/env bash
# SparseDriveV2 一键环境搭建（conda + 依赖 + navsim 包 + CUDA 算子）
#
# 前置：已安装 conda/miniconda，且有 NVIDIA GPU + 与 torch==2.0.1 匹配的 CUDA 驱动。
#   torch==2.0.1 默认对应 CUDA 11.7/11.8 轮子；编译 ops 需要本机有 nvcc(CUDA toolkit)。
#
# 用法：
#   cd <repo>
#   bash setup/setup_env.sh
#   conda activate navsim
#   source setup/env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"
ENV_NAME="navsim"

echo "==> [1/4] 创建 conda 环境 ($ENV_NAME, python=3.9)"
# environment.yml 里声明 name: navsim / python=3.9 / pip 装 requirements.txt
conda env create -f environment.yml || {
    echo "环境可能已存在，改为更新依赖：";
    conda run -n "$ENV_NAME" pip install -r requirements.txt;
}

echo "==> [2/4] 以可编辑模式安装 navsim 包 (setup.py)"
conda run -n "$ENV_NAME" pip install -e .

echo "==> [3/4] 编译可变形聚合 CUDA 算子 (navsim/agents/sparsedrive/ops)"
# 对应 docs/train_eval.md: cd navsim/agents/sparsedrive/ops && python setup.py develop
# 会编译 deformable_aggregation_ext 和 deformable_aggregation_with_depth_ext
( cd navsim/agents/sparsedrive/ops && conda run -n "$ENV_NAME" python setup.py develop )

echo "==> [4/4] 校验关键依赖与算子导入"
conda run -n "$ENV_NAME" python - <<'PY'
import torch, timm, pytorch_lightning, nuplan
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
from navsim.agents.sparsedrive.ops import deformable_aggregation_func  # 触发算子导入
print("deformable_aggregation 算子导入 OK")
PY

cat <<EOF

================ 环境就绪 ================
接下来：
  conda activate $ENV_NAME
  source setup/env.sh          # 设置数据/实验路径环境变量(先按需修改 setup/env.sh)
然后按 setup/prepare_data.sh 准备数据/地图/anchor，再运行训练或推理。
=========================================
EOF
