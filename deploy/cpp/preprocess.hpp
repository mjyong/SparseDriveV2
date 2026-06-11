#pragma once
// SparseDriveV2 预处理：逐式复刻 sparsedrive_features.py 的 test_mode 路径
//   get_camera_params -> resize_crop_flip_img(test) -> normalize_img -> data_adapter
#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace sparsedrive {

constexpr int kNumCams = 3;     // config.cams = (cam_l0, cam_f0, cam_r0)
constexpr int kInH = 256;       // config.final_dim
constexpr int kInW = 512;
constexpr int kNumPoses = 8;    // 输出轨迹点数 (4s @ 0.5s)

struct CameraInput {
  cv::Mat image;                 // 原始图像, BGR (cv::imread 默认) 任意分辨率
  std::array<double, 9> K;       // 内参 3x3, 行优先, 对应原始分辨率
  std::array<double, 9> R;       // sensor2lidar_rotation 3x3 行优先
  std::array<double, 3> t;       // sensor2lidar_translation
};

struct EgoInput {
  std::array<float, 4> driving_command;  // 导航指令 one-hot
  std::array<float, 2> velocity;         // 自车系 (vx, vy) m/s
  std::array<float, 2> acceleration;     // 自车系 (ax, ay) m/s^2
};

struct ModelInputs {
  std::vector<float> imgs;            // (1, 3, 3, 256, 512)
  std::vector<float> projection_mat;  // (1, 3, 4, 4)  增广后的 lidar2img
  std::vector<float> image_wh;        // (1, 3, 2) = (512, 256)
  std::vector<float> status_feature;  // (1, 8)
};

// cams 顺序必须与训练一致: [cam_l0, cam_f0, cam_r0]
ModelInputs preprocess(const std::array<CameraInput, kNumCams>& cams,
                       const EgoInput& ego);

}  // namespace sparsedrive
