# ROS2 集成用法（SparseDriveRosInfer）

在你们 BEVNode 里初始化一次、每帧调用。相机映射：`cam_f0=front, cam_l0=left_front, cam_r0=right_front`。

## 初始化（构造一次）

```cpp
#include "ros2_sparsedrive_infer.hpp"

// 三路标定 [f0, l0, r0]：K=原始分辨率内参, R=sensor2lidar 旋转, t=平移, D=畸变(5或8参,空=不去畸变)
std::array<CameraCalib, 3> calib;
calib[0] = { /*K*/{fx,0,cx, 0,fy,cy, 0,0,1},
             /*R*/{...9...}, /*t*/{...3...}, /*D*/{k1,k2,p1,p2,k3,k4,k5,k6} };  // front
calib[1] = { ... };  // left_front
calib[2] = { ... };  // right_front

bool undistort = true;   // 强畸变相机开；轻畸变可关
auto infer = std::make_shared<SparseDriveRosInfer>(
    "/path/to/sparsedrive_v2_fp32.engine", calib, undistort);
```

> 标定建议直接从你们 `cam_utils_by_json_` 的 JSON 读出来填进 `calib`。

## 每帧调用（在拿到 SensorMsgsWrapper 之后）

```cpp
// msgs 是你们已赋值完成的 SensorMsgsWrapper
auto traj = infer->infer(msgs);   // 8 个 (x,y,heading)，自车系(x前,y左)，0.5s×8=4s

// 加进 foxglove SceneUpdate（与你们地图线同一套实体）
foxglove_msgs::msg::SceneUpdate update;
// ... 你们已有的地图实体 push 进 update.entities ...
infer->addTrajectoryLine(traj, update, /*frame_id=*/"base_link");
scene_pub_->publish(update);
```

## 数据对接点（已按你给的字段写好）

| 模型输入 | 来源 | 处理 |
|---|---|---|
| 3 路图像 | `front/left_front/right_front_image`(YUV422) | `cvtColor(YUV2BGR_YUYV)` → 可选去畸变 → resize/底部crop/归一化 |
| 速度(vx,vy) | `latest_integration_odom_->velocity_body`(RFU) | `rfu_to_flu` → 取前/左两维 |
| 加速度(ax,ay) | `latest_imu_state_->vehicle_linear_acceleration_without_g`(RFU) | `rfu_to_flu` → 取前/左两维 |
| driving_command | 默认直行 `{0,1,0,0}` | ★ one-hot 顺序需核对训练约定 |

## 必须核对的点

- **driving_command one-hot 顺序**：默认 `{0,1,0,0}` 假设 index1=直行。请用一个有缓存的样本反推确认；错了模型会按错误导航意图出轨迹。
- **rfu_to_flu**：内置实现 `{y,-x,z}`（RFU→FLU）。你们已有自己的 `rfu_to_flu`，建议替换 `ros2_sparsedrive_infer.hpp` 里的私有版以保持一致。
- **去畸变 alpha**：默认 0（无黑边、裁少量 FOV）。需保留全 FOV 改 `getOptimalNewCameraMatrix(...,1.0)`。
- **FOV/域差**：模型在 NAVSIM 1920×1080 特定 FOV 上训练，你们相机 FOV/表观不同会有性能损失（需微调/联合训练，见前述）。
- **性能**：本预处理是 CPU(OpenCV)。若不够快，按你们 `ImgPreProcess` 的 CUDA kernel 风格把
  `preprocess.cpp` 的「resize→底部crop→RGB mean/std 归一化→CHW」**精确**搬到 GPU
  （注意必须沿用本仓库的 recipe，不能直接用你们 BEV 模型那套不同的归一化/pad）。
