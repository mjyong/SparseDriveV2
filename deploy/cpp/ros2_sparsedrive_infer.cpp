#include "ros2_sparsedrive_infer.hpp"

#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

SparseDriveRosInfer::SparseDriveRosInfer(const std::string& engine_path,
                                         const std::array<CameraCalib, 3>& calib,
                                         bool undistort)
    : engine_(std::make_unique<TrtEngine>(engine_path)),
      calib_(calib),
      undistort_(undistort) {}

// YUV422(YUYV) -> BGR。data 为借用内存，返回 clone 保证生命周期安全。
cv::Mat SparseDriveRosInfer::decodeYUV422ToBGR(const uint8_t* data, int width, int height) {
  if (!data || width <= 0 || height <= 0)
    throw std::invalid_argument("decodeYUV422ToBGR: 非法图像尺寸/数据");
  cv::Mat yuv422(height, width, CV_8UC2, const_cast<uint8_t*>(data));
  cv::Mat bgr;
  cv::cvtColor(yuv422, bgr, cv::COLOR_YUV2BGR_YUYV);
  return bgr.clone();
}

// 去畸变开关。开时: 首帧懒构建 remap 映射 + K_rect(原始分辨率), 之后 remap; K_out 换成 K_rect。
// 关时: 原样返回, K_out 不变。
cv::Mat SparseDriveRosInfer::maybeUndistort(int cam, const cv::Mat& bgr,
                                            std::array<double, 9>& K_out) {
  if (!undistort_ || calib_[cam].D.empty()) return bgr;

  if (!undistort_ready_[cam]) {
    cv::Mat K(3, 3, CV_64F, calib_[cam].K.data());
    cv::Mat D(static_cast<int>(calib_[cam].D.size()), 1, CV_64F, calib_[cam].D.data());
    cv::Size sz(bgr.cols, bgr.rows);
    // alpha=0: 裁掉无效区无黑边(损失少量 FOV)。D 长度=8 时 cv2 自动用 RATIONAL_MODEL。
    cv::Mat Krect = cv::getOptimalNewCameraMatrix(K, D, sz, 0.0);
    cv::initUndistortRectifyMap(K, D, cv::noArray(), Krect, sz, CV_16SC2, map1_[cam], map2_[cam]);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) K_rect_[cam][i * 3 + j] = Krect.at<double>(i, j);
    undistort_ready_[cam] = true;
  }

  cv::Mat rect;
  cv::remap(bgr, rect, map1_[cam], map2_[cam], cv::INTER_LINEAR);
  K_out = K_rect_[cam];  // 矫正后内参，供 lidar2img 使用
  return rect;
}

void SparseDriveRosInfer::addTrajectoryLine(const std::vector<std::array<float, 3>>& traj,
                                            foxglove_msgs::msg::SceneUpdate& update,
                                            const std::string& frame_id) const {
  foxglove_msgs::msg::SceneEntity ent;
  ent.frame_id = frame_id;        // 自车系: x 前, y 左 (REP-103)，与模型输出一致
  ent.id = "sparsedrive_trajectory";

  foxglove_msgs::msg::LinePrimitive line;
  line.type = foxglove_msgs::msg::LinePrimitive::LINE_STRIP;
  line.pose.orientation.w = 1.0;
  line.thickness = 0.25;          // 米
  line.scale_invariant = false;
  line.color.r = 0.1f;
  line.color.g = 1.0f;
  line.color.b = 0.1f;
  line.color.a = 1.0f;

  line.points.reserve(traj.size() + 1);
  geometry_msgs::msg::Point ego;  // 自车原点起点
  ego.x = 0.0; ego.y = 0.0; ego.z = 0.0;
  line.points.push_back(ego);
  for (const auto& p : traj) {
    geometry_msgs::msg::Point pt;
    pt.x = p[0];
    pt.y = p[1];
    pt.z = 0.0;
    line.points.push_back(pt);
  }

  ent.lines.push_back(line);
  update.entities.push_back(std::move(ent));
}
