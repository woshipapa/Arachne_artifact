#!/bin/bash

# --- 准备环境 ---
# 激活你的 Python 虚拟环境 (conda 或 venv)
# source activate your_env_name

# --- 加速编译的环境变量设置 ---

# 1. 并行编译: 根据你的要求，硬编码为 192 个核心
export MAX_JOBS=16
echo "Using hardcoded MAX_JOBS=${MAX_JOBS}"

# 2. 指定 GPU 架构: Hopper 架构是 9.0
export TORCH_CUDA_ARCH_LIST="9.0"
echo "Building for CUDA arch: ${TORCH_CUDA_ARCH_LIST} (Hopper)"

# 3. 使用 lld 快速链接器 (需要先安装 lld)
export CMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"
export CMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"
export CMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
echo "Using lld linker."
export USE_SYSTEM_NCCL=1
export USE_NCCL=1

export VERBOSE=1
export DEBUG=1
export USE_CUDA=1

# --- 开始编译 ---
echo "Starting PyTorch compilation..."
python setup.py develop

echo "Build finished!"