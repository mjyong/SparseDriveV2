"""
SparseDriveV2 -> ONNX 导出（float32），并与 PyTorch / onnxruntime 做数值对齐校验。

用法（GPU/CPU 均可导出；建议先跑 deploy/test_daf_equivalence.py）:
    python deploy/export_onnx.py \
        --ckpt ckpt/sparsedrive_navsimv2_90p3.ckpt --version v2 \
        --out deploy/sparsedrive_v2_fp32.onnx \
        [--cache-sample exp/data_cache_mini/<log>/<token>]   # 用真实缓存样本校验(推荐)

输出:
    *.onnx            opset16, 输入 imgs/projection_mat/image_wh/status_feature, 输出 trajectory(1,8,3)
    *_io.npz          本次校验用的输入 + PyTorch 输出，供 verify_trt.py 比对 TRT 引擎
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.daf_torch_patch import apply_daf_torch_patch

V1_METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "time_to_collision_within_bound", "comfort", "ego_progress",
)


class TRTSparseDrive(torch.nn.Module):
    """纯张量 IO 封装：dict 进出不被 ONNX 支持，拆成 4 个输入、1 个输出。"""

    def __init__(self, sparsedrive_model):
        super().__init__()
        self.model = sparsedrive_model

    def forward(self, imgs, projection_mat, image_wh, status_feature):
        features = {
            "camera_feature": {
                "imgs": imgs,                      # (B, N, 3, 256, 512) 归一化后图像
                "projection_mat": projection_mat,  # (B, N, 4, 4) lidar2img
                "image_wh": image_wh,              # (B, N, 2)  (W, H)
            },
            "status_feature": status_feature,      # (B, 8)
        }
        output, _ = self.model(features, None)
        return output["trajectory"]                # (B, 8, 3)


def build_agent(ckpt: str, version: str):
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent

    cfg = SparseDriveConfig()
    if version == "v1":
        cfg.dataset_version = "v1"
        cfg.metrics = V1_METRICS
    cfg.velocity_filter_num = (64, 20)             # 与训练/评测脚本覆写保持一致
    agent = SparseDriveAgent(config=cfg, lr=0.0, checkpoint_path=ckpt)
    agent.initialize()
    agent.eval()
    return agent, cfg


def load_inputs_from_cache(token_dir: str, agent):
    """从 data_cache 的某个 token 目录加载真实样本（含完整 test_mode 预处理）。"""
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    builder = agent.get_feature_builders()[0]
    feats = load_feature_target_from_pickle(Path(token_dir) / "sparsedrive_feature.gz")
    feats, _, _ = builder.pipeline(feats, {}, "export", test_mode=True)
    cam = feats["camera_feature"]
    return (
        cam["imgs"][None].float(),
        cam["projection_mat"][None].float(),
        torch.as_tensor(np.ascontiguousarray(cam["image_wh"]))[None].float(),
        feats["status_feature"][None].float(),
    )


def random_inputs(cfg):
    """无数据时的随机输入（仅用于跑通图结构；数值校验请用 --cache-sample）。"""
    N = len(cfg.cams)
    H, W = cfg.final_dim
    from deploy.test_daf_equivalence import build_projection_mat
    return (
        torch.randn(1, N, 3, H, W),
        build_projection_mat(N)[None],
        torch.tensor([[float(W), float(H)]]).expand(N, 2)[None].clone(),
        torch.randn(1, 8),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out", default="deploy/sparsedrive_fp32.onnx")
    ap.add_argument("--cache-sample", default=None, help="data_cache 中某 token 目录，用真实样本校验")
    ap.add_argument("--opset", type=int, default=16)   # GridSample 需要 >=16
    args = ap.parse_args()

    apply_daf_torch_patch()                            # ★ 替换 CUDA 算子，必须在前向之前
    agent, cfg = build_agent(args.ckpt, args.version)
    wrapper = TRTSparseDrive(agent._sparsedrive_model).eval()

    inputs = (
        load_inputs_from_cache(args.cache_sample, agent)
        if args.cache_sample else random_inputs(cfg)
    )

    # 1) PyTorch 基准输出（fp32, CPU 即可——CUDA 算子已被替换）
    with torch.no_grad():
        ref = wrapper(*inputs)
    print(f"[torch] trajectory shape={tuple(ref.shape)}  first pose={ref[0, 0].tolist()}")

    # 2) 导出 ONNX
    torch.onnx.export(
        wrapper, inputs, args.out,
        input_names=["imgs", "projection_mat", "image_wh", "status_feature"],
        output_names=["trajectory"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[onnx] saved -> {args.out}")

    # 3) onnxruntime 数值校验（若已安装）
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            ["trajectory"],
            {k: v.numpy() for k, v in zip(
                ["imgs", "projection_mat", "image_wh", "status_feature"], inputs)},
        )[0]
        diff = np.abs(ort_out - ref.numpy())
        print(f"[verify torch<->ort] max|diff|={diff.max():.3e}  mean={diff.mean():.3e}")
        assert diff.max() < 1e-3, "ONNX 与 PyTorch 输出不一致！"
        print("[PASS] ONNX 与 PyTorch 数值一致")
    except ImportError:
        print("[skip] 未安装 onnxruntime，跳过 ORT 校验 (pip install onnxruntime)")

    # 4) 保存本次 IO，供 verify_trt.py 对照 TRT 引擎
    io_path = str(Path(args.out).with_suffix("")) + "_io.npz"
    np.savez(
        io_path,
        imgs=inputs[0].numpy(), projection_mat=inputs[1].numpy(),
        image_wh=inputs[2].numpy(), status_feature=inputs[3].numpy(),
        trajectory_ref=ref.numpy(),
    )
    print(f"[io] saved -> {io_path}")


if __name__ == "__main__":
    main()
