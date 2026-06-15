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

from deploy.infer import (
    SparseDriveInference,
    cameras_from_agent_input,
    ego_from_agent_input,
    visualize,
)


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
    ap.add_argument("--require-route", action="store_true",
                    help="只保留有 route(roadblock_ids) 的场景；默认不过滤(自采数据通常无 route，"
                         "且 route 不进 SparseDriveV2 网络，仅评测/打分才需要)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 构建模型
    infer = SparseDriveInference(args.ckpt, version=args.version, use_torch_daf=args.cpu_daf)
    cam_names = list(infer.cfg.cams)

    # 2) 构建 SceneLoader 直接读 OpenScene pkl
    #    sensor_config 用 agent 的配置(加载 SparseDrive 需要的相机帧)，保证 .image 被读入
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        has_route=args.require_route,  # 默认 False：不按 route 过滤（自采数据通常无 roadblock_ids）
    )
    scene_loader = SceneLoader(
        data_path=data_root / "navsim_logs" / args.split,
        original_sensor_path=data_root / "sensor_blobs" / args.split,
        scene_filter=scene_filter,
        sensor_config=infer.agent.get_sensor_config(),
    )
    tokens = scene_loader.tokens
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[info] {args.split}: 共 {len(scene_loader.tokens)} 个场景，本次处理 {len(tokens)} 个")

    np.set_printoptions(precision=3, suppress=True)
    for i, token in enumerate(tokens):
        # 3) 取当前帧的相机 + ego 输入
        agent_input = scene_loader.get_agent_input_from_token(token)
        cameras = cameras_from_agent_input(agent_input, cam_names)
        ego = ego_from_agent_input(agent_input)

        # 4) 推理
        traj = infer(cameras, ego)  # (8,3) 自车系

        # 5) 可视化
        png = os.path.join(args.out_dir, f"{token}.png")
        visualize(cameras, traj, save_path=png, ground_z=args.ground_z)
        print(f"[{i + 1}/{len(tokens)}] {token}  endpoint(x,y)={traj[-1, :2]}  -> {png}")

    print(f"[done] 输出在 {args.out_dir}")


if __name__ == "__main__":
    main()
