#!/bin/bash

source ./setup_pyenv.sh
setup_env_and_install

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PYTHONPATH:/root/TensorProbe
export PYTHONPATH=$PYTHONPATH:/root/bucket_config
export USE_FAKE_BATCH=1

export MODEL_TYPE="hunyuan"
export RESOLUTION="720p"
# hunyuan
export NUM_LAYERS=20
export NUM_SINGLE_LAYERS=40
# wan
export NUM_WAN_LAYERS=35
# --- 颜色定义 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- 变量设置 (示例) ---
# 以下配置可以任意组合，以测试其独立性
# export GLOO_TRANSPORT_DEVICE=tcp
# 开启 VAE Cost Model
export VAE_PROFILE_BATCH=0

# 关闭 DiT 逐层 Cost Model
export ENABLE_PROFILE_DIT_LAYER=0

export PER_ITERS=4
export TEST_SP=2
export DELTA_ITER=0
# 关闭数据模拟
export SIMULATE_DATA_ONLY=0

# 禁用通用的 Profiling 任务
export PROFILE_TASK_TYPES=""


export EXPERIMENT=1
export IS_BASELINE=1
echo ""
echo -e "${BLUE}================== 配置检查 ==================${NC}"

# 检查 1: 数据模拟模式
if [ "$SIMULATE_DATA_ONLY" == "1" ]; then
  echo -e "📊 ${GREEN}数据模式: 数据模拟已开启 (模拟生成分桶数据负载)。${NC}"
else
  echo -e "📊 ${YELLOW}数据模式: 数据模拟已禁用 (标准训练模式)。${NC}"
fi

# 检查 2: 通用性能分析任务
if [ -n "$PROFILE_TASK_TYPES" ]; then
  echo -e "⏱️  ${GREEN}通用性能分析: 已启用 (分析目标: $PROFILE_TASK_TYPES)。${NC}"
else
  echo -e "⏱️  ${YELLOW}通用性能分析: 已禁用。${NC}"
fi

# 检查 3: VAE Cost Model
if [ "$VAE_PROFILE_BATCH" == "1" ]; then
  echo -e "📈 ${GREEN}VAE Cost Model: 已开启 (独立性能建模)。${NC}"
else
  echo -e "📈 ${YELLOW}VAE Cost Model: 已禁用。${NC}"
fi

# 检查 4: DiT 逐层建模 (包含其子选项)
if [ "$ENABLE_PROFILE_DIT_LAYER" == "1" ]; then
  # 主选项开启
  echo -e "🧱 ${GREEN}DiT 逐层建模: 已开启 (独立性能建模)。${NC}"
  
  # -- 子选项检查: 仅在主选项开启时显示 --
  # 子选项 4.1: SP 动态调整周期
  if [ -n "$PER_ITERS" ] && [ "$PER_ITERS" -gt 0 ]; then
    echo -e "  🔄 ${GREEN}  - SP动态调整: 已开启 (每 ${PER_ITERS} 次迭代SP将翻倍)。${NC}"
  else
    echo -e "  🔄 ${YELLOW}  - SP动态调整: 已禁用。${NC}"
  fi
  # 子选项 4.2: 起始SP设置
  if [ -n "$TEST_SP" ]; then
    STARTING_SP=$((2**TEST_SP))
    echo -e "  🏁 ${GREEN}  - 起始SP设置: 偏移量 TEST_SP=${TEST_SP} (起始SP = 2^${TEST_SP} => ${STARTING_SP})。${NC}"
  else
    echo -e "  🏁 ${YELLOW}  - 起始SP设置: 未指定偏移量 (默认SP)。${NC}"
  fi

else
  # 主选项禁用，所有相关子选项的日志都不会显示
  echo -e "🧱 ${YELLOW}DiT 逐层建模: 已禁用。${NC}"
fi

# 检查 5: 正式实验模式 (包含基准测试子模式)
if [ "$EXPERIMENT" == "1" ]; then
    # --- EXPERIMENT 已开启，进一步判断是否为 Baseline 模式 ---
    if [ "$IS_BASELINE" == "1" ]; then
        # 基准测试模式
        # rm -rf handler_logs/
        echo -e "🚀 ${GREEN}正式实验模式: 已开启 (Baseline基准测试模式)。${NC}"
        # 计算并显示将要使用的基准配置文件
        BASELINE_SP_VALUE=$((2**TEST_SP))
        echo -e "  🎯 ${GREEN}  - 将使用固定基准任务: vae_tasks_${BASELINE_SP_VALUE}.yml (DP=${BASELINE_SP_VALUE}, SP=${BASELINE_SP_VALUE})。${NC}"
    else
        # 动态调度池模式
        echo -e "🚀 ${GREEN}正式实验模式: 已开启 (将从动态调度池读取任务)。${NC}"
    fi
else
  # --- EXPERIMENT 已禁用 ---
  echo -e "🚀 ${YELLOW}正式实验模式: 已禁用 (使用默认或脚本内定义的任务)。${NC}"
fi


echo -e "${BLUE}==============================================${NC}"
echo "配置检查完毕，准备执行主命令..."
echo ""

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT
# export NCCL_SOCKET_IFNAME=eth0
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'10.244.117.234'}
echo '$MASTER_ADDR' $MASTER_ADDR
MASTER_PORT=${MASTER_PORT:-'12340'}
NNODES=${WORLD_SIZE:-'2'}

echo '$NNODES' $NNODES
NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE
TP=$1
CP=$2
FRAMES=$3
RES_W=$4
RES_H=$5
if [ -n "$6" ]; then
    DEBUG_DIR=$6
fi

MBS=1
# GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
GBS=${GBS:-2}


# CHECKPOINT_PATH=/root/ckpt_tp4_2040_linearparallel_epoch1step2700
# CHECKPOINT_PATH=/root/ckpt_tp${TP}_36_linearparallel_epoch1step2700
# CHECKPOINT_PATH=/root/ckpt_tp2_36_linearparallel
# CHECKPOINT_PATH=/data02/adk/Megatron_VAST/ckpt_tp4_2040
# CHECKPOINT_PATH=/root/ckpt_tp1_36
# CHECKPOINT_PATH=/root/ckpt_tp2_510
# CHECKPOINT_PATH=/root/ckpt_tp1_510
TENSORBOARD_LOGS_PATH=./logs
# VOCAB_FILE=/root/gpt_2_vocab.json
MERGE_FILE=/root/gpt_2_merge.txt
DATA_PATH=./checkpoint


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 1
    --hidden-size 3072        
    --num-attention-heads 24
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
     --global-batch-size ${GBS}
    --train-iters 200
    --weight-decay 1e-2
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5 
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --use-distributed-optimizer
    --recompute-num-layers 42
    --no-rope-fusion
    --distributed-timeout-minutes 60
    # --distribute-saved-activations
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
)
DATA_ARGS=(
    --data-path $DATA_PATH 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type external
    --num-workers 1
    --num-frames ${FRAMES}
    --video-resolution ${RES_W} ${RES_H}
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 100
    --eval-interval 10000 
    # --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    #--pretrained-checkpoint  /root/transformer
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

if [ -n "$DEBUG_DIR" ]; then
    EVAL_AND_LOGGING_ARGS+=(--debug)
    EVAL_AND_LOGGING_ARGS+=(--debug-dir $DEBUG_DIR)
fi

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IBEXT_DISABLE=1
# echo $NCCL_SOCKET_IFNAME
# echo $NCCL_IB_DISABLE
# echo $NCCL_IBEXT_DISABLE
# export NCCL_DEBUG=INFO

# rm test/test_data/tp${TP}cp${CP}_layer36.log
echo start tp${TP} cp${CP} training
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_hunyuanvideo.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} #> test/test_data/tp${TP}cp${CP}_layer36.log
