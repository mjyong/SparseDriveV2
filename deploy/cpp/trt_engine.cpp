#include "trt_engine.hpp"

#include <fstream>
#include <iostream>

using nvinfer1::Dims;
using nvinfer1::TensorIOMode;

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
}

static size_t dimsVolume(const Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
  return v;
}

TrtEngine::TrtEngine(const std::string& engine_path) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
  context_ = engine_->createExecutionContext();
  if (!context_) throw std::runtime_error("createExecutionContext failed");
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // 静态 shape 引擎：按引擎记录的张量形状一次性分配显存并绑定地址
  int nb = engine_->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char* name = engine_->getIOTensorName(i);
    Dims dims = engine_->getTensorShape(name);
    size_t vol = dimsVolume(dims);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, vol * sizeof(float)));
    dev_buf_[name] = ptr;
    vol_[name] = vol;
    if (!context_->setTensorAddress(name, ptr))
      throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
  }
}

TrtEngine::~TrtEngine() {
  for (auto& kv : dev_buf_) cudaFree(kv.second);
  if (stream_) cudaStreamDestroy(stream_);
  delete context_;
  delete engine_;
  delete runtime_;
}

size_t TrtEngine::tensorVolume(const std::string& name) const {
  auto it = vol_.find(name);
  if (it == vol_.end()) throw std::runtime_error("unknown tensor: " + name);
  return it->second;
}

void TrtEngine::setInput(const std::string& name, const float* host_data,
                         size_t count) {
  if (count != tensorVolume(name))
    throw std::runtime_error("size mismatch for tensor '" + name + "': got " +
                             std::to_string(count) + ", expect " +
                             std::to_string(tensorVolume(name)));
  CUDA_CHECK(cudaMemcpyAsync(dev_buf_.at(name), host_data,
                             count * sizeof(float), cudaMemcpyHostToDevice,
                             stream_));
}

void TrtEngine::infer() {
  if (!context_->enqueueV3(stream_))
    throw std::runtime_error("enqueueV3 failed");
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

std::vector<float> TrtEngine::getOutput(const std::string& name) {
  std::vector<float> out(tensorVolume(name));
  CUDA_CHECK(cudaMemcpy(out.data(), dev_buf_.at(name),
                        out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return out;
}
