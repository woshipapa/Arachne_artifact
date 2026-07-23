#!/bin/bash

# source activate your_env_name


export MAX_JOBS=16
echo "Using hardcoded MAX_JOBS=${MAX_JOBS}"

export TORCH_CUDA_ARCH_LIST="9.0"
echo "Building for CUDA arch: ${TORCH_CUDA_ARCH_LIST} (Hopper)"

export CMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"
export CMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"
export CMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
echo "Using lld linker."
export USE_SYSTEM_NCCL=1
export USE_NCCL=1

export VERBOSE=1
export DEBUG=1
export USE_CUDA=1

echo "Starting PyTorch compilation..."
python setup.py develop

echo "Build finished!"