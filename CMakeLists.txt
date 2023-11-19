cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sparse_test)

set(CMAKE_PREFIX_PATH "/opt/libtorch")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")

link_directories(/usr/local/cuda-12.1/lib64)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT ON)

find_package(Torch REQUIRED)

add_executable(sparse_test src/main.cpp)
target_link_libraries(sparse_test "${TORCH_LIBRARIES}" CUDA::cublas CUDA::cublasLt CUDA::cufft)
set_property(TARGET sparse_test PROPERTY CXX_STANDARD 17)
