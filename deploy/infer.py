"""
SparseDriveV2 端到端推理（standalone）：原始图像 + 标定 + can_bus -> 轨迹。

不依赖 navsim 的 Scene/AgentInput/缓存；只复用 SparseDriveFeatureBuilder 的真实
预处理方法（resize/crop、归一化、lidar2img 构建），把"从磁盘读图"替换为接受内存图像，
其余每一步与官方 eval 的 pipeline(test_mode=True) 完全一致，保证几何/数值对齐。

输入约定
--------
cameras: dict, 键为 config.cams 里的相机名（默认 cam_l0/cam_f0/cam_r0），每个值:
    {
      "image": np.ndarray (H,W,3) RGB uint8        # 二选一
      "image_path": str                            # 或给文件路径
      "intrinsics": (3,3)              相机内参 K（对应 image 的原始分辨率）
      "sensor2lidar_rotation": (3,3)   相机->LiDAR 旋转
      "sensor2lidar_translation": (3,) 相机->LiDAR 平移
      "distortion": (5,) 可选，默认 0
    }
ego: dict, 从 can_bus 抽取：
    {
      "driving_command": (4,)  导航指令 one-hot（如 [left,straight,right,unknown]）
      "velocity":        (2,)  自车系速度 (vx, vy) [m/s]
      "acceleration":    (2,)  自车系加速度 (ax, ay) [m/s^2]
    }

输出
----
trajectory: np.ndarray (8,3) = (x, y, heading)，自车(后轴)系，0.5s 间隔、4s 时域。

依赖文件（构建模型时需就位，相对仓库根）
    ckpt/resnet34.bin
    ckpt/kmeans/{path_1024.npy, velocity_256.npy, trajectory_1024_256.npz}
    <你的权重>.ckpt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# v1 用 6 指标头；v2 用默认 8 指标头
V1_METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "time_to_collision_within_bound", "comfort", "ego_progress",
)


class SparseDriveInference:
    def __init__(self, ckpt: str, version: str = "v2", device: str = None, use_torch_daf: bool = False):
        """
        :param ckpt: 模型权重 .ckpt
        :param version: "v1" / "v2"（影响指标头与打分公式）
        :param device: "cuda"/"cpu"；默认自动
        :param use_torch_daf: True 时用 grid_sample 等价实现替换 CUDA 算子，可在 CPU 上跑
        """
        if use_torch_daf:
            from deploy.daf_torch_patch import apply_daf_torch_patch
            apply_daf_torch_patch()

        from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
        from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent

        cfg = SparseDriveConfig()
        if version == "v1":
            cfg.dataset_version = "v1"
            cfg.metrics = V1_METRICS
        cfg.velocity_filter_num = (64, 20)          # 与训练/评测脚本覆写保持一致
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = SparseDriveAgent(config=cfg, lr=0.0, checkpoint_path=ckpt)
        self.agent.initialize()                      # 加载权重
        self.agent = self.agent.to(self.device).eval()
        self.builder = self.agent.get_feature_builders()[0]

    # ---------- 预处理：复用 builder 的真实方法，仅替换读图步骤 ----------
    def _build_frame_info(self, cameras: dict) -> dict:
        info = {}
        for name in self.cfg.cams:
            c = cameras[name]
            info[name] = {
                "image_path": c.get("image_path"),
                "sensor2lidar_rotation": np.asarray(c["sensor2lidar_rotation"], np.float64),
                "sensor2lidar_translation": np.asarray(c["sensor2lidar_translation"], np.float64),
                "intrinsics": np.asarray(c["intrinsics"], np.float64),
                "distortion": np.asarray(c.get("distortion", np.zeros(5)), np.float64),
            }
        return info

    @torch.no_grad()
    def preprocess(self, cameras: dict, ego: dict) -> dict:
        b = self.builder

        # 1) 由标定构建 lidar2img / 内参（builder 原生方法，无差异）
        frame_info = self._build_frame_info(cameras)
        results = b.get_camera_params(frame_info)

        # 2) 取图像（内存或磁盘），镜像 builder.load_images 的 BGR 处理
        imgs = []
        for name in self.cfg.cams:
            c = cameras[name]
            img = np.asarray(c["image"]) if c.get("image") is not None \
                else np.array(Image.open(str(c["image_path"])))
            imgs.append(img)
        if b._config.to_bgr:
            import cv2
            imgs = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs]

        # ★ 关键：resize_crop 用 config.H/W 当“原图尺寸”假设，且内参按 resize 缩放，
        #   两者必须一致。这里按实际输入分辨率设 H/W，使任意分辨率都几何自洽。
        h0, w0 = imgs[0].shape[:2]
        b._config.H, b._config.W = int(h0), int(w0)
        results["imgs"] = imgs
        results["img_shape"] = [x.shape[:2] for x in imgs]

        # 3) 其余步骤与 pipeline(test_mode=True) 完全相同
        results = b.resize_crop_flip_img(results, test_mode=True)   # 确定性 resize+中心 crop，同步改内参/lidar2img
        results, _ = b.ego_rotation(results, {}, test_mode=True)    # test_mode 下 no-op
        results = b.photo_metric_distortion(results, test_mode=True)  # test_mode 下 no-op
        results = b.normalize_img(results)                          # 减均值除方差
        results = b.data_adapter(results)                           # 转张量 + projection_mat/image_wh

        # 4) ego 状态：driving_command(4) + velocity(2) + acceleration(2) = 8
        status = torch.tensor(
            [*np.asarray(ego["driving_command"], np.float32),
             *np.asarray(ego["velocity"], np.float32),
             *np.asarray(ego["acceleration"], np.float32)],
            dtype=torch.float32,
        )
        assert status.numel() == 8, f"status_feature 应为 8 维(4+2+2)，得到 {status.numel()}"

        feats = {"camera_feature": results, "status_feature": status}
        return self._batchify(feats)

    def _batchify(self, v):
        if torch.is_tensor(v):
            return v.unsqueeze(0).to(self.device)
        if isinstance(v, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(self.device)
        if isinstance(v, dict):
            return {k: self._batchify(x) for k, x in v.items()}
        return [v]   # img_shape 等列表：模型不用，包成长度1的 batch

    # ---------- 前向 ----------
    @torch.no_grad()
    def __call__(self, cameras: dict, ego: dict) -> np.ndarray:
        batch = self.preprocess(cameras, ego)
        output, _ = self.agent.forward(batch, None)        # (output, loss_dict)
        return output["trajectory"].squeeze(0).cpu().numpy()   # (8,3)


# ---------- 工具：自车系轨迹 -> 全局系（可选，需当前帧全局位姿）----------
def trajectory_to_global(traj: np.ndarray, ego_xy, ego_heading) -> np.ndarray:
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    R = np.array([[c, -s], [s, c]])
    xy = traj[:, :2] @ R.T + np.asarray(ego_xy)
    head = traj[:, 2] + ego_heading
    return np.concatenate([xy, head[:, None]], axis=1)


# ---------- 可验证 demo：从 data_cache 的一个 token 读“原始”输入，跑标准化推理 ----------
def _inputs_from_cache(token_dir: str, cfg):
    """从 data_cache 的 sparsedrive_feature.gz 还原 cameras/ego 原始输入，用于自检。"""
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    feats = load_feature_target_from_pickle(Path(token_dir) / "sparsedrive_feature.gz")
    frame = feats["camera_feature"][-1]            # 最后一帧的原始相机字典
    cameras = {
        name: {
            "image_path": frame[name]["image_path"],
            "intrinsics": frame[name]["intrinsics"],
            "sensor2lidar_rotation": frame[name]["sensor2lidar_rotation"],
            "sensor2lidar_translation": frame[name]["sensor2lidar_translation"],
            "distortion": frame[name]["distortion"],
        }
        for name in cfg.cams
    }
    sf = feats["status_feature"].numpy()
    ego = {"driving_command": sf[:4], "velocity": sf[4:6], "acceleration": sf[6:8]}
    return cameras, ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--cache-token", default=None, help="data_cache 中某 token 目录，跑可验证 demo")
    ap.add_argument("--cpu-daf", action="store_true", help="用 grid_sample 等价算子(可 CPU)")
    args = ap.parse_args()

    infer = SparseDriveInference(args.ckpt, version=args.version, use_torch_daf=args.cpu_daf)

    if args.cache_token:
        cameras, ego = _inputs_from_cache(args.cache_token, infer.cfg)
    else:
        raise SystemExit(
            "请用 --cache-token 跑 demo，或在代码中按文档构造 cameras/ego 字典后调用 infer(cameras, ego)"
        )

    traj = infer(cameras, ego)
    np.set_printoptions(precision=3, suppress=True)
    print("trajectory (x, y, heading) ego-frame:\n", traj)
    print("shape:", traj.shape)


if __name__ == "__main__":
    main()
