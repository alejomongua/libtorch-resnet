# Torch model executed in C++ with LibTorch

This is a simple example of how to use a Torch model in C++ with LibTorch.

## Requirements

- CMake
- LibTorch
- CUDA 12.1
    - cuBLAS
    - cuDNN
    - cuFFT
    - cuSPARSE

## Build

```bash
mkdir build
cd build
cmake ..
make
```