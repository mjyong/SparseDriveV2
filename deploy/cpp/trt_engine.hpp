#pragma once
// 通用 TensorRT 引擎封装（name-based IO API, 兼容 TRT 8.5 ~ 10.x / Thor）
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(err__));                    \
  } while (0)

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // 输入张量元素数（用于上层校验）
  size_t tensorVolume(const std::string& name) const;
  // 拷贝 host fp32 数据到对应输入的显存
  void setInput(const std::string& name, const float* host_data, size_t count);
  // 执行推理（同步）
  void infer();
  // 取输出（D2H 拷贝）
  std::vector<float> getOutput(const std::string& name);

 private:
  TrtLogger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::unordered_map<std::string, void*> dev_buf_;
  std::unordered_map<std::string, size_t> vol_;  // 元素数(fp32)
};
