cmake_minimum_required(VERSION 3.16)


project(mixformer-onnx)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
find_package(CUDA REQUIRED)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)

  # tensorrt
  include_directories(/usr/include/aarch64-linux-gnu)
  link_libraries(/usr/lib/aarch64-linux-gnu)
else()
  message("-- embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/opt/TensorRT/include)
  link_directories(/opt/TensorRT/lib)

endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/include/opencv4)
 
# add_executable(mixformer-onnx main.cpp mixformer_onnx.cpp mixformer_onnx.h)
# target_link_libraries(mixformer-onnx libonnxruntime.so libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_video.so libopencv_videoio.so libopencv_imgcodecs.so)

add_executable(ortracker-trt main_trt.cpp ortrack_trt.cpp ortrack_trt.h logging.h half.hpp)
target_link_libraries(ortracker-trt nvinfer cudart libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_video.so libopencv_videoio.so libopencv_imgcodecs.so)

