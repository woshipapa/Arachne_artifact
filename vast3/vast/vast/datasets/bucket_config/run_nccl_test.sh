#!/bin/bash

# ========== NCCL 环境变量调优 ==========
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1         # 禁用 InfiniBand（有时会崩）
export NCCL_P2P_DISABLE=1        # 禁用P2P做排查
export NCCL_SOCKET_IFNAME=lo     # 绑定回环网卡用于单机调试

# 防止意外挂起
export OMP_NUM_THREADS=1


# ========== 启动参数 ==========
NUM_GPUS=8

echo "Launching NCCL AllReduce test on ${NUM_GPUS} GPUs..."

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  test_dist.py