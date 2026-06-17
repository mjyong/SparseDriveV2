#pragma once
// SparseDriveRosInfer::infer 的模板实现（按字段名从你们的 SensorMsgsWrapper 取数据）。
#include <stdexcept>

template <class WrapperT>
std::vector<std::array<float, 3>> SparseDriveRosInfer::infer(const WrapperT& msgs) {
  // 1) 取 3 路图 (front/left_front/right_front) 并解码为 BGR
  const auto& f = msgs.front_image;
  const auto& l = msgs.left_front_image;
  const auto& r = msgs.right_front_image;
  if (!f || !l || !r) throw std::invalid_argument("infer: 前/左前/右前 图像有空指针");

  auto decode = [](const auto& img) {
    return SparseDriveRosInfer::decodeYUV422ToBGR(
        img->data.data(), static_cast<int>(img->width), static_cast<int>(img->height));
  };
  std::array<cv::Mat, 3> bgr = {decode(f), decode(l), decode(r)};

  // 2) 组装 3 路 CameraInput (去畸变开关 + 标定)
  std::array<sparsedrive::CameraInput, 3> cams;  // [f0, l0, r0]
  for (int n = 0; n < 3; ++n) {
    std::array<double, 9> K = calib_[n].K;
    cams[n].image = maybeUndistort(n, bgr[n], K);  // undistort_ 关时原样返回, K 不变
    cams[n].K = K;                                  // 开时已换成 K_rect
    cams[n].R = calib_[n].R;
    cams[n].t = calib_[n].t;
  }

  // 3) ego: driving_command(默认直行) + 速度(FLU 前/左) + 加速度(FLU 前/左)
  sparsedrive::EgoInput ego;
  ego.driving_command = default_driving_command_;

  const auto& v = msgs.latest_integration_odom_->velocity_body;  // RFU
  auto vel_flu = rfu_to_flu(v.x, v.y, v.z);
  ego.velocity = {static_cast<float>(vel_flu[0]), static_cast<float>(vel_flu[1])};

  const auto& a = msgs.latest_imu_state_->vehicle_linear_acceleration_without_g;  // RFU
  auto acc_flu = rfu_to_flu(a.x, a.y, a.z);
  ego.acceleration = {static_cast<float>(acc_flu[0]), static_cast<float>(acc_flu[1])};

  // 4) 预处理(与 python 对齐) -> TRT 推理
  sparsedrive::ModelInputs in = sparsedrive::preprocess(cams, ego);
  engine_->setInput("imgs", in.imgs.data(), in.imgs.size());
  engine_->setInput("projection_mat", in.projection_mat.data(), in.projection_mat.size());
  engine_->setInput("image_wh", in.image_wh.data(), in.image_wh.size());
  engine_->setInput("status_feature", in.status_feature.data(), in.status_feature.size());
  engine_->infer();
  std::vector<float> out = engine_->getOutput("trajectory");  // (1,8,3)

  // 5) 整理成 8 个 (x,y,heading)
  std::vector<std::array<float, 3>> traj(sparsedrive::kNumPoses);
  for (int i = 0; i < sparsedrive::kNumPoses; ++i)
    traj[i] = {out[i * 3 + 0], out[i * 3 + 1], out[i * 3 + 2]};
  return traj;
}
