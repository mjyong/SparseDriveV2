#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sparsedrive {

// 归一化参数 (config.img_mean / img_std, RGB 顺序, to_bgr=False)
static const float kMean[3] = {123.675f, 116.28f, 103.53f};
static const float kStd[3] = {58.395f, 57.12f, 57.375f};

using Mat4 = std::array<double, 16>;  // 行优先 4x4

static Mat4 matmul4(const Mat4& a, const Mat4& b) {
  Mat4 c{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += a[i * 4 + k] * b[k * 4 + j];
      c[i * 4 + j] = s;
    }
  return c;
}

// === sparsedrive_features.py:get_camera_params 的逐式复刻 ===
// python:  lidar2cam_r = inv(R);  lidar2cam_rt[:3,:3] = lidar2cam_r.T;
//          lidar2cam_rt[3,:3] = -(t @ lidar2cam_r.T);
//          lidar2img = viewpad @ lidar2cam_rt.T
// 数学化简(R 为正交旋转): lidar2img = K_pad @ [R^T | -R^T t; 0 0 0 1]
static Mat4 buildLidar2Img(const std::array<double, 9>& K,
                           const std::array<double, 9>& R,
                           const std::array<double, 3>& t) {
  Mat4 ext{};  // [R^T | -R^T t]
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) ext[i * 4 + j] = R[j * 3 + i];  // R^T
    double s = 0;
    for (int j = 0; j < 3; ++j) s += R[j * 3 + i] * t[j];       // (R^T t)_i
    ext[i * 4 + 3] = -s;
  }
  ext[15] = 1.0;

  Mat4 viewpad{};  // K 嵌入 4x4
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) viewpad[i * 4 + j] = K[i * 3 + j];
  viewpad[15] = 1.0;

  return matmul4(viewpad, ext);
}

ModelInputs preprocess(const std::array<CameraInput, kNumCams>& cams,
                       const EgoInput& ego) {
  ModelInputs out;
  out.imgs.resize(size_t(1) * kNumCams * 3 * kInH * kInW);
  out.projection_mat.resize(size_t(1) * kNumCams * 16);
  out.image_wh.resize(size_t(1) * kNumCams * 2);
  out.status_feature.resize(8);

  for (int n = 0; n < kNumCams; ++n) {
    const CameraInput& cam = cams[n];
    if (cam.image.empty()) throw std::runtime_error("empty image at cam " + std::to_string(n));
    const int H0 = cam.image.rows, W0 = cam.image.cols;

    // === resize_crop_flip_img (test_mode=True) 的逐式复刻 ===
    // resize = max(fH/H, fW/W); newW=int(W*r); newH=int(H*r)
    // crop_h = newH - fH (bot_pct_lim=(0,0) -> 保留图像底部 256 行)
    // crop_w = (newW - fW) / 2 (水平居中)
    const double r = std::max(double(kInH) / H0, double(kInW) / W0);
    const int newW = int(W0 * r), newH = int(H0 * r);
    const int crop_w = int(std::max(0, newW - kInW) / 2);
    const int crop_h = newH - kInH;
    if (crop_h < 0)
      throw std::runtime_error("image too small after resize (cam " + std::to_string(n) + ")");

    cv::Mat resized;
    // PIL.Image.resize 的默认插值随 Pillow 版本而异(新版 BICUBIC)。这里用 INTER_CUBIC，
    // 与 Python 侧可能有 <1 像素级的细微差异 —— 用 --raw 模式可绕过预处理做逐位对齐验证。
    cv::resize(cam.image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_CUBIC);
    cv::Mat cropped = resized(cv::Rect(crop_w, crop_h, kInW, kInH));

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);  // 模型吃 RGB (to_bgr=False)

    // normalize + HWC->CHW
    float* dst = out.imgs.data() + size_t(n) * 3 * kInH * kInW;
    for (int y = 0; y < kInH; ++y) {
      const uint8_t* row = rgb.ptr<uint8_t>(y);
      for (int x = 0; x < kInW; ++x)
        for (int c = 0; c < 3; ++c)
          dst[(size_t(c) * kInH + y) * kInW + x] =
              (float(row[x * 3 + c]) - kMean[c]) / kStd[c];
    }

    // === lidar2img 同步增广: extend_matrix @ lidar2img (与 _img_transform 一致) ===
    // python: transform3x3 = [[r,0,-crop_w],[0,r,-crop_h],[0,0,1]] 嵌入 4x4 的 [:3,:3]。
    // 平移分量位于 (0,2)/(1,2) —— 作用在齐次像素 (u*d, v*d, d) 上: new_u*d = d*(r*u - crop_w)
    Mat4 base = buildLidar2Img(cam.K, cam.R, cam.t);
    Mat4 aug{};
    aug[0] = r;   aug[2] = -double(crop_w);
    aug[5] = r;   aug[6] = -double(crop_h);
    aug[10] = 1.0; aug[15] = 1.0;
    Mat4 final_mat = matmul4(aug, base);

    for (int i = 0; i < 16; ++i)
      out.projection_mat[size_t(n) * 16 + i] = float(final_mat[i]);

    out.image_wh[n * 2 + 0] = float(kInW);
    out.image_wh[n * 2 + 1] = float(kInH);
  }

  // status_feature = driving_command(4) + velocity(2) + acceleration(2)
  for (int i = 0; i < 4; ++i) out.status_feature[i] = ego.driving_command[i];
  out.status_feature[4] = ego.velocity[0];
  out.status_feature[5] = ego.velocity[1];
  out.status_feature[6] = ego.acceleration[0];
  out.status_feature[7] = ego.acceleration[1];

  return out;
}

}  // namespace sparsedrive
