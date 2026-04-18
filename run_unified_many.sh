#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Node rank must be provided as the first argument."
    exit 1
fi
export NODE_RANK=$1

source ./setup_pyenv.sh
setup_env_and_install

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
export MAX_FRAMES="129"
export NUM_LAYERS=20
export NUM_SINGLE_LAYERS=40
export NUM_WAN_LAYERS=30
export NUM_COG_LAYERS=30
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

export VAE_PROFILE_BATCH=0

export ENABLE_PROFILE_DIT_LAYER=0

export PER_ITERS=4
export TEST_SP=2
export DELTA_ITER=52
export SIMULATE_DATA_ONLY=0

export PROFILE_TASK_TYPES=""


export EXPERIMENT=1
export IS_BASELINE=0
export SCHEDULE_TYPE="genetic"  

export ENABLE_TIMER=1

export ENABLE_BUCKET_GRAD_OVERLAP=${ENABLE_BUCKET_GRAD_OVERLAP:-0}
export ENABLE_MULTI_BROKER=${ENABLE_MULTI_BROKER:-1}
export ENABLE_REMOTE_GET_PREFETCH=${ENABLE_REMOTE_GET_PREFETCH:-1}
export REMOTE_GET_PREFETCH_LOOKAHEAD=${REMOTE_GET_PREFETCH_LOOKAHEAD:-2}
export REMOTE_GET_PREFETCH_MAX_INFLIGHT=${REMOTE_GET_PREFETCH_MAX_INFLIGHT:-1}
export REMOTE_GET_PREFETCH_WAIT_TIMEOUT=${REMOTE_GET_PREFETCH_WAIT_TIMEOUT:-120.0}
export CUSTOM_GROUP_CACHE_LOOKAHEAD=${CUSTOM_GROUP_CACHE_LOOKAHEAD:-3}
export CUSTOM_GROUP_GC_INTERVAL=${CUSTOM_GROUP_GC_INTERVAL:-3}
export CUSTOM_GROUP_CACHE_MAX_GROUPS=${CUSTOM_GROUP_CACHE_MAX_GROUPS:-16}
export CUSTOM_GROUP_CACHE_MAX_LOCAL_PER_RANK=${CUSTOM_GROUP_CACHE_MAX_LOCAL_PER_RANK:-4}


echo ""
echo -e "${BLUE}================== 配置检查 (Node Rank: $NODE_RANK) ==================${NC}"

if [ "$SIMULATE_DATA_ONLY" == "1" ]; then
  echo -e "📊 ${GREEN}数据模式: 数据模拟已开启 (模拟生成分桶数据负载)。${NC}"
else
  echo -e "📊 ${YELLOW}数据模式: 数据模拟已禁用 (标准训练模式)。${NC}"
fi

if [ -n "$PROFILE_TASK_TYPES" ]; then
  echo -e "⏱️  ${GREEN}通用性能分析: 已启用 (分析目标: $PROFILE_TASK_TYPES)。${NC}"
else
  echo -e "⏱️  ${YELLOW}通用性能分析: 已禁用。${NC}"
fi

if [ "$VAE_PROFILE_BATCH" == "1" ]; then
  echo -e "📈 ${GREEN}VAE Cost Model: 已开启 (独立性能建模)。${NC}"
else
  echo -e "📈 ${YELLOW}VAE Cost Model: 已禁用。${NC}"
fi

if [ "$ENABLE_PROFILE_DIT_LAYER" == "1" ]; then
  echo -e "🧱 ${GREEN}DiT 逐层建模: 已开启 (独立性能建模)。${NC}"
  
  if [ -n "$PER_ITERS" ] && [ "$PER_ITERS" -gt 0 ]; then
    echo -e "  🔄 ${GREEN}  - SP动态调整: 已开启 (每 ${PER_ITERS} 次迭代SP将翻倍)。${NC}"
  else
    echo -e "  🔄 ${YELLOW}  - SP动态调整: 已禁用。${NC}"
  fi
  if [ -n "$TEST_SP" ]; then
    STARTING_SP=$((2**TEST_SP))
    echo -e "  🏁 ${GREEN}  - 起始SP设置: 偏移量 TEST_SP=${TEST_SP} (起始SP = 2^${TEST_SP} => ${STARTING_SP})。${NC}"
  else
    echo -e "  🏁 ${YELLOW}  - 起始SP设置: 未指定偏移量 (默认SP)。${NC}"
  fi

else
  echo -e "🧱 ${YELLOW}DiT 逐层建模: 已禁用。${NC}"
fi

if [ "$EXPERIMENT" == "1" ]; then
    if [ "$IS_BASELINE" == "1" ]; then
        echo -e "🚀 ${GREEN}正式实验模式: 已开启 (Baseline基准测试模式)。${NC}"
        BASELINE_SP_VALUE=$((2**TEST_SP))
        echo -e "  🎯 ${GREEN}  - 将使用固定基准任务: vae_tasks_${BASELINE_SP_VALUE}.yml (DP=${BASELINE_SP_VALUE}, SP=${BASELINE_SP_VALUE})。${NC}"
    else
        echo -e "🚀 ${GREEN}正式实验模式: 已开启 (将从动态调度池读取任务)。${NC}"
    fi
else
  echo -e "🚀 ${YELLOW}正式实验模式: 已禁用 (使用默认或脚本内定义的任务)。${NC}"
fi




echo -e ""
echo -e "${BLUE}------------------ 模型与任务配置 ------------------${NC}"

echo -e "🧠 ${GREEN}模型类型:${NC} ${MODEL_TYPE}"
echo -e "🖼️  ${GREEN}分辨率:${NC} ${RESOLUTION}"

if [ "$MODEL_TYPE" == "hunyuan" ]; then
  echo -e "  📐 ${GREEN}Hunyuan DiT层数:${NC} NUM_LAYERS=${NUM_LAYERS}, NUM_SINGLE_LAYERS=${NUM_SINGLE_LAYERS}"
elif [ "$MODEL_TYPE" == "wan" ]; then
  echo -e "  📐 ${GREEN}WAN Transformer层数:${NC} NUM_WAN_LAYERS=${NUM_WAN_LAYERS}"
elif [ "$MODEL_TYPE" == "cogvideox" ]; then
  echo -e "  📐 ${GREEN}CogVideoX层数:${NC} NUM_COG_LAYERS=${NUM_COG_LAYERS}"
else
  echo -e "  📐 ${YELLOW}未知模型类型:${NC} ${MODEL_TYPE}"
fi


echo -e ""
echo -e "${BLUE}------------------ 调度与系统策略 ------------------${NC}"

echo -e "🗂️  ${GREEN}调度策略 (SCHEDULE_TYPE):${NC} ${SCHEDULE_TYPE}"

case "$SCHEDULE_TYPE" in
  megatron-lm)
    echo -e "  🔹 ${GREEN}使用 Megatron-LM 朴素 DP×SP 静态划分策略${NC}"
    ;;
  flex_sp)
    echo -e "  🔹 ${GREEN}使用 FlexSP 动态调度策略${NC}"
    ;;
  genetic)
    echo -e "  🧬 ${GREEN}使用 Genetic Search 调度策略${NC}"
    ;;
  *)
    echo -e "  ⚠️  ${YELLOW}未知调度策略，请确认实现是否存在${NC}"
    ;;
esac

if [ "$ENABLE_TIMER" == "1" ]; then
  echo -e "⏲️  ${GREEN}系统 Timer:${NC} 已开启（关键路径计时）"
else
  echo -e "⏲️  ${YELLOW}系统 Timer:${NC} 已关闭"
fi

echo -e ""
echo -e "${BLUE}------------------ 通信优化配置 ------------------${NC}"

if [ "$ENABLE_MULTI_BROKER" == "1" ]; then
  echo -e "🧭 ${GREEN}Multi Broker:${NC} 已开启（每节点 Broker + source 路由）"
else
  echo -e "🧭 ${YELLOW}Multi Broker:${NC} 已关闭（回退单 Broker=rank0）"
fi

if [ "$ENABLE_REMOTE_GET_PREFETCH" == "1" ]; then
  echo -e "🚚 ${GREEN}Remote GET Prefetch:${NC} 已开启"
  echo -e "  🔭 ${GREEN}  - LOOKAHEAD:${NC} ${REMOTE_GET_PREFETCH_LOOKAHEAD}"
  echo -e "  🧵 ${GREEN}  - MAX_INFLIGHT:${NC} ${REMOTE_GET_PREFETCH_MAX_INFLIGHT}"
  echo -e "  ⏱️  ${GREEN}  - WAIT_TIMEOUT:${NC} ${REMOTE_GET_PREFETCH_WAIT_TIMEOUT}"
else
  echo -e "🚚 ${YELLOW}Remote GET Prefetch:${NC} 已关闭"
fi

echo -e "🗃️  ${GREEN}Custom Group Cache:${NC} LOOKAHEAD=${CUSTOM_GROUP_CACHE_LOOKAHEAD}, GC_INTERVAL=${CUSTOM_GROUP_GC_INTERVAL}, MAX_GROUPS=${CUSTOM_GROUP_CACHE_MAX_GROUPS}, MAX_LOCAL_PER_RANK=${CUSTOM_GROUP_CACHE_MAX_LOCAL_PER_RANK}"


echo -e ""
echo -e "${BLUE}------------------ CUDA / PyTorch 配置 ------------------${NC}"

echo -e "🧩 ${GREEN}CUDA_VISIBLE_DEVICES:${NC} ${CUDA_VISIBLE_DEVICES}"
echo -e "🔗 ${GREEN}CUDA_DEVICE_MAX_CONNECTIONS:${NC} ${CUDA_DEVICE_MAX_CONNECTIONS}"

if [ "$NVTE_FLASH_ATTN" == "1" ]; then
  echo -e "⚡ ${GREEN}NVTE FlashAttention:${NC} 启用"
else
  echo -e "⚡ ${YELLOW}NVTE FlashAttention:${NC} 禁用"
fi

if [ "$NVTE_FUSED_ATTN" == "1" ]; then
  echo -e "🧬 ${GREEN}NVTE Fused Attention:${NC} 启用"
else
  echo -e "🧬 ${YELLOW}NVTE Fused Attention:${NC} 禁用"
fi

echo -e "🧠 ${GREEN}PyTorch CUDA Allocator:${NC} ${PYTORCH_CUDA_ALLOC_CONF}"


echo -e ""
echo -e "${BLUE}------------------ 数据 / 调试模式 ------------------${NC}"

if [ "$USE_FAKE_BATCH" == "1" ]; then
  echo -e "🧪 ${GREEN}Fake Batch:${NC} 已启用（使用模拟 batch）"
else
  echo -e "🧪 ${YELLOW}Fake Batch:${NC} 已禁用（真实数据路径）"
fi

if [ -n "$DEBUG_DIR" ]; then
  echo -e "🐞 ${GREEN}Debug 模式:${NC} 已开启"
  echo -e "  📁 Debug 输出目录: ${DEBUG_DIR}"
else
  echo -e "🐞 ${YELLOW}Debug 模式:${NC} 未开启"
fi



echo -e "${BLUE}==============================================${NC}"
echo "配置检查完毕，准备执行主命令..."
echo ""

GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

MASTER_ADDR=${MASTER_ADDR:-'10.244.15.102'}
echo '$MASTER_ADDR' $MASTER_ADDR
MASTER_PORT=${MASTER_PORT:-'12344'}
NNODES=${WORLD_SIZE:-'2'}

echo '$NNODES' $NNODES
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE

TP=$2
CP=$3
FRAMES=$4
RES_W=$5
RES_H=$6
if [ -n "$7" ]; then
    DEBUG_DIR=$7
fi

MBS=1
GBS=${GBS:-2}

TENSORBOARD_LOGS_PATH=./logs
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
    --train-iters 1000
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
    --distributed-timeout-minutes 1
)

if [ "${ENABLE_BUCKET_GRAD_OVERLAP}" == "1" ]; then
    TRAINING_ARGS+=(--force-bucketing)
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
)
DATA_ARGS=(
    --data-path $DATA_PATH 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
    --num-frames ${FRAMES}
    --video-resolution ${RES_W} ${RES_H}
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 100
    --eval-interval 10000 
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

if [ -n "$DEBUG_DIR" ]; then
    EVAL_AND_LOGGING_ARGS+=(--debug)
    EVAL_AND_LOGGING_ARGS+=(--debug-dir $DEBUG_DIR)
fi

echo start tp${TP} cp${CP} training
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_hunyuanvideo.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]}

exit_code=$?
echo "torchrun exited with code $exit_code"
exit $exit_code
