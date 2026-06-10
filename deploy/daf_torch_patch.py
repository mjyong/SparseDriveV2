"""
用纯 PyTorch (F.grid_sample) 等价替换自定义 CUDA 算子 deformable_aggregation_ext，
使 SparseDriveV2 可导出 ONNX / 转 TensorRT。

等价性依据（对照 ops/src/deformable_aggregation_cuda.cu 逐条复刻）：
  1. 采样坐标:  h_im = loc_h * H - 0.5, w_im = loc_w * W - 0.5  (.cu:174-175)
     == F.grid_sample(align_corners=False) 的坐标约定 (grid = 2*loc - 1)
  2. 双线性插值四角越界置零 (.cu:30-49)
     == padding_mode="zeros"
  3. 整个相机贡献由 loc∈(0,1) 门控 (.cu:166)
     == 显式 valid mask 相乘
  4. 加权求和: 每点特征按 (cam, level, group) 权重加权后对 cam/level 求和 (.cu:161-186)
     == 分组 reshape 后乘 weights 再 sum

用法：
    from deploy.daf_torch_patch import apply_daf_torch_patch
    apply_daf_torch_patch()      # 必须在构建模型之前或之后均可（类级 monkeypatch）
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _daf_forward_torch(
    self,
    instance_feature: torch.Tensor,   # (B, A, C)
    anchor: torch.Tensor,             # (B, A, num_sample*2) 拉平的 (x,y) 序列
    anchor_embed,                     # None (本仓库调用方式)
    feature_maps,                     # deformable_format 的产物，此实现不用（改用 metas["feature_maps"]）
    metas: dict,                      # camera_feature dict: feature_maps / projection_mat / image_wh
    depth_prob=None,
    **kwargs,
):
    """DeformableFeatureAggregation.forward 的 ONNX 可导出等价实现。"""
    assert depth_prob is None, "本仓库 decoder 不使用 depth 分支"
    bs, num_anchor = instance_feature.shape[:2]

    # 1) 3D 关键点（与原实现共用同一模块，无差异）
    key_points = self.kps_generator(anchor, instance_feature)        # (B, A, P, 3)

    # 2) 投影到各相机（与原实现共用 project_points，无差异）
    points_2d, depth, mask = self.project_points(
        key_points, metas["projection_mat"], metas.get("image_wh")
    )  # points_2d: (B, N, A, P, 2) 归一化[0,1]; mask: (B, N, A, P)

    # 3) softmax 采样权重（与原实现共用 _get_weights，无差异）
    weights = self._get_weights(instance_feature, anchor_embed, metas, mask)
    # weights: (B, A, N, L, P, G)

    fms = metas["feature_maps"]                                       # list[L] of (B, N, C, H, W)
    num_cams = fms[0].shape[1]
    num_levels = len(fms)
    C = fms[0].shape[2]

    # 4) grid_sample 采样 —— 对应 .cu 的 bilinear_sampling
    #    grid 约定: (x=w方向, y=h方向)，与 points_2d 的 (u, v) 一致
    grid = points_2d * 2.0 - 1.0                                      # (B, N, A, P, 2)
    grid = grid.flatten(0, 1)                                         # (B*N, A, P, 2)
    sampled = []
    for fm in fms:
        s = F.grid_sample(
            fm.flatten(0, 1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )                                                             # (B*N, C, A, P)
        sampled.append(s)
    feats = torch.stack(sampled, dim=1)                               # (B*N, L, C, A, P)
    feats = feats.reshape(bs, num_cams, num_levels, C, num_anchor, -1)
    feats = feats.permute(0, 4, 1, 2, 5, 3)                           # (B, A, N, L, P, C)

    # 5) loc∈(0,1) 门控 —— 对应 .cu:166 的 if 整段跳过
    valid = (
        (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1)
        & (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1)
    )                                                                 # (B, N, A, P)
    valid = valid.permute(0, 2, 1, 3)[:, :, :, None, :, None]         # (B, A, N, 1, P, 1)
    feats = feats * valid.to(feats.dtype)

    # 6) 分组加权 + 对 (cam, level) 求和 —— 对应 .cu 的 result 累加
    G = self.num_groups
    feats = feats.reshape(*feats.shape[:-1], G, C // G)               # (B, A, N, L, P, G, C/G)
    feats = (feats * weights[..., None]).sum(dim=(2, 3))              # (B, A, P, G, C/G)
    feats = feats.reshape(bs, num_anchor, -1, C)                      # (B, A, P, C)

    # 7) 对采样点求和 + 输出投影 + 残差（与原 forward 尾部一致）
    feats = feats.sum(dim=2)                                          # (B, A, C)
    output = self.proj_drop(self.output_proj(feats))
    if self.residual_mode == "add":
        output = output + instance_feature
    elif self.residual_mode == "cat":
        output = torch.cat([output, instance_feature], dim=-1)
    return output


def apply_daf_torch_patch():
    """把 DeformableFeatureAggregation.forward 替换为 ONNX 可导出实现（类级，全局生效）。"""
    from navsim.agents.sparsedrive.blocks import DeformableFeatureAggregation
    DeformableFeatureAggregation._forward_cuda_original = DeformableFeatureAggregation.forward
    DeformableFeatureAggregation.forward = _daf_forward_torch
    return DeformableFeatureAggregation
