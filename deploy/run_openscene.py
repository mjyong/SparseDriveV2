"""
直接加载“已转成 OpenScene 标准格式 + pkl”的自采数据，跑 SparseDriveV2 推理并可视化。

前提：你的数据按 NAVSIM/OpenScene 布局摆好（与官方一致）：
    <DATA_ROOT>/navsim_logs/<split>/<log_name>.pkl     # 每条 log 一个 pkl(scene_dict_list)
    <DATA_ROOT>/sensor_blobs/<split>/<log>/<cam>/...    # 图像 blob，pkl 里 data_path 指向它
pkl 里每帧 scene_dict 必须含 SparseDriveV2 用到的字段：
    cams[cam_name] = {data_path, sensor2lidar_rotation, sensor2lidar_translation,
                      cam_intrinsic, distortion}        # 见 navsim Cameras.from_camera_dict
    ego2global_translation / ego2global_rotation, ego_dynamic_state(速度/加速度),
    driving_command, token, timestamp, roadblock_ids(可选, has_route 用)

用法:
    python deploy/run_openscene.py \
        --ckpt ckpt/sparsedrive_navsimv2_90p3.ckpt --version v2 \
        --data-root /path/to/your_dataset --split my_split \
        --out-dir exp/openscene_vis [--limit 20] [--ground-z 0.0] [--cpu-daf] [--require-route]

输出: 每个 token 一张 <token>.png（3 相机+前视投影轨迹+BEV），并打印轨迹。
注: route(roadblock_ids) 不进 SparseDriveV2 网络，默认不按 route 过滤场景；
    仅当你要复现官方"只评测有 route 的场景"时才加 --require-route。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

from deploy.infer import SparseDriveInference, rectify_image, visualize


def load_inputs_direct(scene_loader, token, cam_names, num_history_frames, sensor_blobs_path,
                       undistort: bool = False, undistort_alpha: float = 0.0):
    """
    直接从已解析的原始 frame 字典读取 3 路相机 + ego，绕过 Cameras.from_camera_dict
    的“必须含 8 路相机”硬编码假设（自采数据常只有 SparseDrive 用的 3 路）。
    字段名与 navsim AgentInput.from_scene_dict_list 完全一致。
    :param undistort: True 时把图像去畸变成针孔图并替换为矫正后内参（强畸变相机必用）
    """
    from PIL import Image
    frames = scene_loader.scene_frames_dicts[token]      # 原始 frame_list
    cur = frames[num_history_frames - 1]                 # 当前帧
    cam_dict = {k.lower(): v for k, v in cur["cams"].items()}  # 兼容大小写键名

    cameras = {}
    for name in cam_names:
        if name not in cam_dict:
            raise KeyError(f"pkl 缺相机 '{name}'，当前帧仅有: {list(cam_dict.keys())}")
        c = cam_dict[name]
        K = np.asarray(c["cam_intrinsic"], np.float64)
        D = np.asarray(c.get("distortion", np.zeros(5)), np.float64)
        entry = {
            "intrinsics": K,
            "sensor2lidar_rotation": np.asarray(c["sensor2lidar_rotation"], np.float64),
            "sensor2lidar_translation": np.asarray(c["sensor2lidar_translation"], np.float64),
            "distortion": D,
        }
        if undistort:
            img = np.array(Image.open(str(Path(sensor_blobs_path) / c["data_path"])))  # RGB
            img_rect, K_rect = rectify_image(img, K, D, alpha=undistort_alpha)
            entry["image"] = img_rect          # 矫正后图像(数组)
            entry["intrinsics"] = K_rect        # 矫正后内参
            entry["distortion"] = np.zeros(5)   # 已矫正，后续按无畸变处理
        else:
            entry["image_path"] = Path(sensor_blobs_path) / c["data_path"]
        cameras[name] = entry

    eds = cur["ego_dynamic_state"]                       # [vx, vy, ax, ay]
    ego = {
        "driving_command": np.asarray(cur["driving_command"], np.float32),
        "velocity": np.asarray(eds[:2], np.float32),
        "acceleration": np.asarray(eds[2:], np.float32),
    }
    return cameras, ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--data-root", required=True, help="含 navsim_logs/ 与 sensor_blobs/ 的根目录")
    ap.add_argument("--split", default="my_split", help="navsim_logs/<split> 与 sensor_blobs/<split>")
    ap.add_argument("--out-dir", default="exp/openscene_vis")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个场景")
    ap.add_argument("--ground-z", type=float, default=0.0)
    ap.add_argument("--cpu-daf", action="store_true")
    ap.add_argument("--undistort", action="store_true",
                    help="喂网络前去畸变(强畸变/广角相机必开；模型是纯针孔投影)")
    ap.add_argument("--undistort-alpha", type=float, default=0.0,
                    help="0=裁掉无效区无黑边(默认), 1=保留全部像素有黑边")
    ap.add_argument("--require-route", action="store_true",
                    help="只保留有 route(roadblock_ids) 的场景；默认不过滤(自采数据通常无 route，"
                         "且 route 不进 SparseDriveV2 网络，仅评测/打分才需要)")
    # ---- 场景/token 选择 ----
    ap.add_argument("--log-names", nargs="+", default=None,
                    help="只加载这些 pkl(不带 .pkl 后缀)，如 --log-names log_a log_b")
    ap.add_argument("--tokens", nargs="+", default=None, help="只加载这些场景 token")
    ap.add_argument("--tokens-file", default=None, help="从文本文件读 token(每行一个)")
    ap.add_argument("--max-scenes", type=int, default=None, help="最多加载几个场景")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 构建模型
    infer = SparseDriveInference(args.ckpt, version=args.version, use_torch_daf=args.cpu_daf)
    cam_names = list(infer.cfg.cams)

    # 2) 构建 SceneLoader 直接读 OpenScene pkl
    #    sensor_config 用 agent 的配置(加载 SparseDrive 需要的相机帧)，保证 .image 被读入
    # token 选择：命令行 --tokens 或 --tokens-file(每行一个)
    sel_tokens = args.tokens
    if args.tokens_file:
        with open(args.tokens_file) as f:
            sel_tokens = [t.strip() for t in f if t.strip()]

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        has_route=args.require_route,  # 默认 False：不按 route 过滤（自采数据通常无 roadblock_ids）
        log_names=args.log_names,      # None=全部 pkl；否则只加载这些(不带 .pkl)
        tokens=sel_tokens,             # None=全部 token；否则只加载这些
        max_scenes=args.max_scenes,    # None=不限
    )
    scene_loader = SceneLoader(
        data_path=data_root / "navsim_logs" / args.split,
        original_sensor_path=data_root / "sensor_blobs" / args.split,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),  # 不用 AgentInput，直接读原始 dict
    )
    if len(scene_loader.tokens) == 0:
        raise SystemExit(
            "加载到 0 个场景。可能原因：log-names/tokens 写错、数据字段缺失、"
            "或 has_route 过滤(自采数据加默认即可，不要传 --require-route)。"
        )
    tokens = scene_loader.tokens
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[info] {args.split}: 共 {len(scene_loader.tokens)} 个场景，本次处理 {len(tokens)} 个")

    sensor_blobs_path = data_root / "sensor_blobs" / args.split
    np.set_printoptions(precision=3, suppress=True)
    for i, token in enumerate(tokens):
        # 3) 取当前帧的相机 + ego 输入（直接读原始 dict，不走 8 路硬编码的 AgentInput）
        cameras, ego = load_inputs_direct(
            scene_loader, token, cam_names,
            scene_filter.num_history_frames, sensor_blobs_path,
            undistort=args.undistort, undistort_alpha=args.undistort_alpha,
        )

        # 4) 推理
        traj = infer(cameras, ego)  # (8,3) 自车系

        # 5) 可视化
        png = os.path.join(args.out_dir, f"{token}.png")
        visualize(cameras, traj, save_path=png, ground_z=args.ground_z)
        print(f"[{i + 1}/{len(tokens)}] {token}  endpoint(x,y)={traj[-1, :2]}  -> {png}")

    print(f"[done] 输出在 {args.out_dir}")


if __name__ == "__main__":
    main()
