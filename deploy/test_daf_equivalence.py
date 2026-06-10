"""
算子级等价性验证：CUDA deformable_aggregation_ext  vs  grid_sample 等价实现。
必须在 GPU 机器（已编译 ops）上运行：
    python deploy/test_daf_equivalence.py
判定标准：trajectory 级联前的逐元素 max|diff| < 1e-4 (float32 数值噪声水平)。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.blocks import DeformableFeatureAggregation
from navsim.agents.sparsedrive.ops import deformable_format
from deploy.daf_torch_patch import _daf_forward_torch


def build_projection_mat(num_cams=3):
    """构造一个前向针孔相机的 lidar2img（与 feature builder 同款拼法），保证大量点投影有效。"""
    mats = []
    yaw_list = [0.55, 0.0, -0.55][:num_cams]                      # 左/前/右
    K = np.array([[400.0, 0, 256, 0], [0, 400.0, 128, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    for yaw in yaw_list:
        # ego(x前,y左,z上) -> cam(x右,y下,z前)
        c, s = np.cos(yaw), np.sin(yaw)
        R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])      # 相机朝向绕 z 旋转
        ego2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) @ R_ego.T
        rt = np.eye(4)
        rt[:3, :3] = ego2cam
        rt[:3, 3] = ego2cam @ np.array([0.0, 0.0, -1.6])          # 相机高 1.6m
        mats.append(K @ rt)
    return torch.tensor(np.stack(mats), dtype=torch.float32)


def main():
    assert torch.cuda.is_available(), "此测试需要 GPU（对照 CUDA 算子）"
    torch.manual_seed(0)
    device = "cuda"
    cfg = SparseDriveConfig()

    B, A, C, N, L = 2, 16, 256, 3, 4
    daf = DeformableFeatureAggregation(
        config=cfg, embed_dims=C, num_groups=8, num_levels=L, num_cams=N,
        num_pts=cfg.len_path, attn_drop=0.0,
        use_deformable_func=True, use_camera_embed=True, residual_mode="add",
    ).to(device).eval()

    # 多尺度特征图 (与 backbone 输出同构)
    shapes = [(64, 128), (32, 64), (16, 32), (8, 16)]
    fms = [torch.randn(B, N, C, h, w, device=device) for h, w in shapes]
    deform_value = deformable_format(fms)

    # anchor: 自车前方的路径点 (x∈[2,40], y∈[-8,8])，拉平成 (B,A,len_path*2)
    pts = torch.rand(B, A, cfg.len_path, 2, device=device)
    pts[..., 0] = pts[..., 0] * 38 + 2
    pts[..., 1] = (pts[..., 1] - 0.5) * 16
    anchor = pts.flatten(-2)

    inst = torch.randn(B, A, C, device=device)
    metas = {
        "feature_maps": fms,
        "projection_mat": build_projection_mat(N)[None].repeat(B, 1, 1, 1).to(device),
        "image_wh": torch.tensor([[512.0, 256.0]], device=device).expand(B * N, 2).reshape(B, N, 2),
    }

    with torch.no_grad():
        out_cuda = DeformableFeatureAggregation.forward(daf, inst, anchor, None, deform_value, metas, None)
        out_torch = _daf_forward_torch(daf, inst, anchor, None, deform_value, metas, None)

    diff = (out_cuda - out_torch).abs()
    rel = diff.max() / out_cuda.abs().max().clamp(min=1e-9)
    print(f"max|diff| = {diff.max().item():.3e}   mean|diff| = {diff.mean().item():.3e}   rel = {rel.item():.3e}")
    assert diff.max().item() < 1e-4, "CUDA 与 PyTorch 实现不等价！请勿继续导出。"
    print("[PASS] CUDA DAF 与 grid_sample 等价实现数值一致 (fp32)")


if __name__ == "__main__":
    main()
