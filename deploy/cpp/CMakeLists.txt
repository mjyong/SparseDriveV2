cmake_minimum_required(VERSION 3.18)
project(sparsedrive_trt LANGUAGES CXX)

# 目标环境: NVIDIA Thor (aarch64, CUDA 12.x, TensorRT 10.x)
# 同样兼容 x86 + TensorRT >= 8.5 (代码只用 name-based IO API: setTensorAddress/enqueueV3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT: Thor/Jetson 上在系统路径; 自定义安装用 -DTENSORRT_ROOT=/path/to/TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include
  PATH_SUFFIXES include)
find_library(TENSORRT_LIB nvinfer
  HINTS ${TENSORRT_ROOT} /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib:     ${TENSORRT_LIB}")

add_executable(sparsedrive_infer
  main.cpp
  trt_engine.cpp
  preprocess.cpp)

target_include_directories(sparsedrive_infer PRIVATE ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sparsedrive_infer PRIVATE
  ${TENSORRT_LIB}
  CUDA::cudart
  ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# ROS2 集成节点 (ros2_sparsedrive_infer.cpp) 不在此独立 CMake 构建，
# 而是放到你们的 ROS2 workspace(ament_cmake) 里编译，因为它依赖你们的消息包
# (seres_camera_msgs / foxglove_msgs / geometry_msgs 等)。集成示例:
#   add_library(sparsedrive_ros ros2_sparsedrive_infer.cpp preprocess.cpp trt_engine.cpp)
#   ament_target_dependencies(sparsedrive_ros rclcpp foxglove_msgs geometry_msgs
#                             seres_camera_msgs seres_map_location_msgs msg_interfaces)
#   target_link_libraries(sparsedrive_ros ${TENSORRT_LIB} CUDA::cudart ${OpenCV_LIBS})

