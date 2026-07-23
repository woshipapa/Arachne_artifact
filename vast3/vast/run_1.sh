
#! /bin/bash


# source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=$(/root.sh) #必须有，不修改，本次升级改动
# echo NCCL_IB_HCA=$NCCL_IB_HCA


export PYTHONPATH=$PYTHONPATH:/root/deepspeed_baseline/vast3/accelerate/src/
export PYTHONPATH=$PYTHONPATH:/root/deepspeed_baseline/vast3/vast
export WORLD_SIZE=2
export RANK=1
# git clone -b anon/dev <internal-git-repo> && (cd accelerate && curl -kLo `git rev-parse --git-dir`/hooks/commit-msg <internal-git-host> chmod +x `git rev-parse --git-dir`/hooks/commit-msg)

# cd accelerate
# pip install -e .
# cd ..
# export NCCL_NVLS_ENABLE=0&&export NCCL_CROSS_NIC=0&&export NCCL_ALGO=RING&&python tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy.py

# export NCCL_NVLS_ENABLE=0&&export NCCL_CROSS_NIC=0&&export NCCL_ALGO=RING&&
# python tools/train.py projects/cogvideox/configs/cogvideox_i2v.py
# hunyuan
python tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy.py 
# wan
# python tools/train.py projects/wan/configs/wan_i2vhy.py