// SparseDriveV2 TensorRT 部署入口 (NVIDIA Thor / 任意 TRT>=8.5 环境)
//
// 模式 1: 完整管线 (图像 + 标定 + can_bus -> 轨迹)
//   ./sparsedrive_infer <engine> --calib calib.yaml
//
// 模式 2: raw 对齐验证 (绕过预处理, 读 python export 的输入张量, 输出应与
//         deploy/sparsedrive_*_io.npz 中 trajectory_ref 一致)
//   ./sparsedrive_infer <engine> --raw <dir>   # dir 含 imgs.bin 等 4 个 fp32 文件
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "preprocess.hpp"
#include "trt_engine.hpp"

using sparsedrive::CameraInput;
using sparsedrive::EgoInput;
using sparsedrive::kNumCams;
using sparsedrive::kNumPoses;

static std::vector<float> readRawF32(const std::string& path, size_t expect) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<float> v(expect);
  f.read(reinterpret_cast<char*>(v.data()), expect * sizeof(float));
  if (size_t(f.gcount()) != expect * sizeof(float))
    throw std::runtime_error("size mismatch reading " + path);
  return v;
}

template <size_t N>
static std::array<double, N> matToArr(const cv::Mat& m) {
  cv::Mat d;
  m.convertTo(d, CV_64F);
  std::array<double, N> a{};
  std::memcpy(a.data(), d.ptr<double>(), N * sizeof(double));
  return a;
}

static void runAndPrint(TrtEngine& engine, const std::vector<float>& imgs,
                        const std::vector<float>& proj,
                        const std::vector<float>& wh,
                        const std::vector<float>& status) {
  engine.setInput("imgs", imgs.data(), imgs.size());
  engine.setInput("projection_mat", proj.data(), proj.size());
  engine.setInput("image_wh", wh.data(), wh.size());
  engine.setInput("status_feature", status.data(), status.size());
  engine.infer();
  std::vector<float> traj = engine.getOutput("trajectory");  // (1, 8, 3)

  std::cout << "trajectory (x, y, heading) ego-frame:\n";
  for (int i = 0; i < kNumPoses; ++i)
    std::printf("  t=%.1fs  %8.3f %8.3f %8.4f\n", 0.5 * (i + 1),
                traj[i * 3 + 0], traj[i * 3 + 1], traj[i * 3 + 2]);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage:\n  " << argv[0] << " <engine> --calib calib.yaml\n  "
              << argv[0] << " <engine> --raw <dir>\n";
    return 1;
  }
  try {
    TrtEngine engine(argv[1]);
    const std::string mode = argv[2];

    if (mode == "--raw") {
      const std::string dir = argv[3];
      runAndPrint(engine,
                  readRawF32(dir + "/imgs.bin", engine.tensorVolume("imgs")),
                  readRawF32(dir + "/projection_mat.bin", engine.tensorVolume("projection_mat")),
                  readRawF32(dir + "/image_wh.bin", engine.tensorVolume("image_wh")),
                  readRawF32(dir + "/status_feature.bin", engine.tensorVolume("status_feature")));
      return 0;
    }

    if (mode != "--calib") throw std::runtime_error("unknown mode: " + mode);

    // ---- 解析 calib.yaml (OpenCV FileStorage) ----
    cv::FileStorage fs(argv[3], cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error(std::string("cannot open ") + argv[3]);

    static const char* kCamNames[kNumCams] = {"cam_l0", "cam_f0", "cam_r0"};
    std::array<CameraInput, kNumCams> cams;
    for (int n = 0; n < kNumCams; ++n) {
      const std::string base = kCamNames[n];
      std::string img_path;
      fs[base + "_image"] >> img_path;
      cams[n].image = cv::imread(img_path, cv::IMREAD_COLOR);
      if (cams[n].image.empty()) throw std::runtime_error("cannot read image " + img_path);
      cv::Mat K, R, t;
      fs[base + "_K"] >> K;
      fs[base + "_R"] >> R;  // sensor2lidar_rotation
      fs[base + "_t"] >> t;  // sensor2lidar_translation
      cams[n].K = matToArr<9>(K);
      cams[n].R = matToArr<9>(R);
      cams[n].t = matToArr<3>(t);
    }

    EgoInput ego{};
    std::vector<float> status;
    fs["status"] >> status;  // 8 维: command(4)+vel(2)+acc(2)
    if (status.size() != 8) throw std::runtime_error("status must have 8 values");
    std::memcpy(ego.driving_command.data(), status.data(), 4 * sizeof(float));
    std::memcpy(ego.velocity.data(), status.data() + 4, 2 * sizeof(float));
    std::memcpy(ego.acceleration.data(), status.data() + 6, 2 * sizeof(float));

    auto in = sparsedrive::preprocess(cams, ego);
    runAndPrint(engine, in.imgs, in.projection_mat, in.image_wh, in.status_feature);
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
