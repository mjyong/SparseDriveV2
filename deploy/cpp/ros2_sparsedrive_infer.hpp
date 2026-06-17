#pragma once
// SparseDriveV2 ROS2 集成推理：SensorMsgsWrapper -> 轨迹 -> foxglove SceneUpdate 线
//
// 复用 deploy/cpp 的 preprocess(与 python 对齐的 resize/crop/归一化/lidar2img) 与 TRT 引擎。
// 相机映射: cam_f0=front, cam_l0=left_front, cam_r0=right_front (SparseDrive 用前/左前/右前 3 路)。
// 注意: 本文件依赖你们的 ROS2 消息类型，需在你们的 ROS 工作区里编译。
#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <foxglove_msgs/msg/scene_update.hpp>
#include <foxglove_msgs/msg/scene_entity.hpp>
#include <foxglove_msgs/msg/line_primitive.hpp>

#include "preprocess.hpp"   // sparsedrive::CameraInput/EgoInput/preprocess
#include "trt_engine.hpp"

// 单路相机标定（原始分辨率内参 + sensor2lidar 外参 + 畸变）
struct CameraCalib {
  std::array<double, 9> K{};   // intrinsics 3x3 行优先
  std::array<double, 9> R{};   // sensor2lidar_rotation 3x3 行优先
  std::array<double, 3> t{};   // sensor2lidar_translation
  std::vector<double> D{};     // 畸变 (5 或 8 参; 空=不畸变)
};

// 与你们 SensorMsgsWrapper 对接所需的最小字段视图（避免直接耦合全部头文件）。
// 用法见 .cpp：直接传你们的 wrapper 即可（模板，按字段名取）。
class SparseDriveRosInfer {
 public:
  // calib: [cam_f0, cam_l0, cam_r0] 三路标定; undistort: 去畸变开关
  SparseDriveRosInfer(const std::string& engine_path,
                      const std::array<CameraCalib, 3>& calib,
                      bool undistort = false);

  // 主流程: 从 wrapper 取 3 路图 + ego -> 推理 -> 返回 8 个 (x,y,heading) 自车系轨迹点
  template <class WrapperT>
  std::vector<std::array<float, 3>> infer(const WrapperT& msgs);

  // 把轨迹加成一条 foxglove 线 (LINE_STRIP), 追加到 update
  void addTrajectoryLine(const std::vector<std::array<float, 3>>& traj,
                         foxglove_msgs::msg::SceneUpdate& update,
                         const std::string& frame_id = "base_link") const;

 private:
  // YUV422(YUYV) -> BGR (preprocess 内部再 BGR->RGB)
  static cv::Mat decodeYUV422ToBGR(const uint8_t* data, int width, int height);

  // 懒构建去畸变映射(首帧拿到尺寸时), 之后 remap; 返回矫正图并把 K 换成 K_rect
  cv::Mat maybeUndistort(int cam, const cv::Mat& bgr, std::array<double, 9>& K_out);

  // RFU(右前上) -> FLU(前左上): {y, -x, z}; 你们若有自己的 rfu_to_flu, 可替换
  static std::array<double, 3> rfu_to_flu(double x, double y, double z) {
    return {y, -x, z};
  }

  std::unique_ptr<TrtEngine> engine_;
  std::array<CameraCalib, 3> calib_;
  bool undistort_;

  // 去畸变缓存(每路一份 remap map + K_rect)
  std::array<cv::Mat, 3> map1_, map2_;
  std::array<std::array<double, 9>, 3> K_rect_{};
  std::array<bool, 3> undistort_ready_{{false, false, false}};

  // driving_command 默认直行。★ one-hot 顺序需按你们训练约定核对(见 .cpp 注释)。
  std::array<float, 4> default_driving_command_{{0.f, 1.f, 0.f, 0.f}};
};

#include "ros2_sparsedrive_infer_impl.hpp"  // 模板 infer() 实现
