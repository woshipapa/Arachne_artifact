#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Node rank must be provided as the first argument."
    exit 1
fi
export NODE_RANK=$1

source "$(dirname "${BASH_SOURCE[0]}")/setup_pyenv.sh" || {
    echo "Error: environment setup failed; aborting before launch."
    exit 1
}

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/my_utils
export USE_FAKE_BATCH=1

export MODEL_TYPE="${MODEL_TYPE:-hunyuan}"
export RESOLUTION="${RESOLUTION:-720p}"
# Frame budget per resolution: 1080p uses 57, others 129. Override MAX_FRAMES to force.
if [ -z "${MAX_FRAMES:-}" ]; then
    if [ "$RESOLUTION" == "1080p" ]; then
        export MAX_FRAMES="57"
    else
        export MAX_FRAMES="129"
    fi
fi
export NUM_LAYERS=${NUM_LAYERS:-20}
export NUM_SINGLE_LAYERS=${NUM_SINGLE_LAYERS:-40}
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
# Plan-index offset: curr_plan = (iteration + DELTA_ITER) / PER_ITERS.
# 0 = fresh run starting from schedule_0 (what the shipped logs used);
# override only when manually resuming into the middle of a plan set.
export DELTA_ITER=${DELTA_ITER:-0}
export SIMULATE_DATA_ONLY=0

export PROFILE_TASK_TYPES=""


export EXPERIMENT=1
export IS_BASELINE=0
export SCHEDULE_TYPE="${SCHEDULE_TYPE:-genetic}"

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
echo -e "${BLUE}================== Config check (Node Rank: $NODE_RANK) ==================${NC}"

if [ "$SIMULATE_DATA_ONLY" == "1" ]; then
  echo -e "📊 ${GREEN}Data mode: simulation ON (synthetic bucketed workloads).${NC}"
else
  echo -e "📊 ${YELLOW}Data mode: simulation OFF (standard training).${NC}"
fi

if [ -n "$PROFILE_TASK_TYPES" ]; then
  echo -e "⏱️  ${GREEN}Profiling: ON (targets: $PROFILE_TASK_TYPES).${NC}"
else
  echo -e "⏱️  ${YELLOW}Profiling: OFF.${NC}"
fi

if [ "$VAE_PROFILE_BATCH" == "1" ]; then
  echo -e "📈 ${GREEN}VAE cost model: ON (standalone profiling).${NC}"
else
  echo -e "📈 ${YELLOW}VAE cost model: OFF.${NC}"
fi

if [ "$ENABLE_PROFILE_DIT_LAYER" == "1" ]; then
  echo -e "🧱 ${GREEN}DiT per-layer profiling: ON.${NC}"
  
  if [ -n "$PER_ITERS" ] && [ "$PER_ITERS" -gt 0 ]; then
    echo -e "  🔄 ${GREEN}  - SP sweep: ON (SP doubles every ${PER_ITERS} iterations).${NC}"
  else
    echo -e "  🔄 ${YELLOW}  - SP sweep: OFF.${NC}"
  fi
  if [ -n "$TEST_SP" ]; then
    STARTING_SP=$((2**TEST_SP))
    echo -e "  🏁 ${GREEN}  - Starting SP: TEST_SP=${TEST_SP} (start SP = 2^${TEST_SP} => ${STARTING_SP}).${NC}"
  else
    echo -e "  🏁 ${YELLOW}  - Starting SP: not set (default).${NC}"
  fi

else
  echo -e "🧱 ${YELLOW}DiT per-layer profiling: OFF.${NC}"
fi

if [ "$EXPERIMENT" == "1" ]; then
    if [ "$IS_BASELINE" == "1" ]; then
        echo -e "🚀 ${GREEN}Experiment mode: ON (static baseline).${NC}"
        BASELINE_SP_VALUE=$((2**TEST_SP))
        echo -e "  🎯 ${GREEN}  - Fixed baseline tasks: vae_tasks_${BASELINE_SP_VALUE}.yml (DP=${BASELINE_SP_VALUE}, SP=${BASELINE_SP_VALUE}).${NC}"
    else
        echo -e "🚀 ${GREEN}Experiment mode: ON (tasks come from the pre-generated plan set).${NC}"
    fi
else
  echo -e "🚀 ${YELLOW}Experiment mode: OFF (default in-script tasks).${NC}"
fi


echo -e ""
echo -e "${BLUE}------------------ Model & task config ------------------${NC}"

echo -e "🧠 ${GREEN}Model:${NC} ${MODEL_TYPE}"
echo -e "🖼️  ${GREEN}Resolution:${NC} ${RESOLUTION}"

if [ "$MODEL_TYPE" == "hunyuan" ]; then
  echo -e "  📐 ${GREEN}Hunyuan DiT layers:${NC} NUM_LAYERS=${NUM_LAYERS}, NUM_SINGLE_LAYERS=${NUM_SINGLE_LAYERS}"
elif [ "$MODEL_TYPE" == "wan" ]; then
  echo -e "  📐 ${GREEN}Wan transformer layers:${NC} NUM_WAN_LAYERS=${NUM_WAN_LAYERS}"
elif [ "$MODEL_TYPE" == "cogvideox" ]; then
  echo -e "  📐 ${GREEN}CogVideoX layers:${NC} NUM_COG_LAYERS=${NUM_COG_LAYERS}"
else
  echo -e "  📐 ${YELLOW}Unknown MODEL_TYPE:${NC} ${MODEL_TYPE}"
fi


echo -e ""
echo -e "${BLUE}------------------ Scheduling & system policy ------------------${NC}"

echo -e "🗂️  ${GREEN}Schedule type:${NC} ${SCHEDULE_TYPE}"

case "$SCHEDULE_TYPE" in
  megatron-lm)
    echo -e "  🔹 ${GREEN}Megatron-LM static DPxSP partitioning${NC}"
    ;;
  flex_sp)
    echo -e "  🔹 ${GREEN}FlexSP dynamic scheduling${NC}"
    ;;
  genetic)
    echo -e "  🧬 ${GREEN}Genetic-search scheduling (Arachne)${NC}"
    ;;
  *)
    echo -e "  ⚠️  ${YELLOW}Unknown schedule type -- check the implementation${NC}"
    ;;
esac

if [ "$ENABLE_TIMER" == "1" ]; then
  echo -e "⏲️  ${GREEN}System timer:${NC} ON (critical-path timing)"
else
  echo -e "⏲️  ${YELLOW}System timer:${NC} OFF"
fi

echo -e ""
echo -e "${BLUE}------------------ Communication config ------------------${NC}"

if [ "$ENABLE_MULTI_BROKER" == "1" ]; then
  echo -e "🧭 ${GREEN}Multi broker:${NC} ON (per-node broker + source routing)"
else
  echo -e "🧭 ${YELLOW}Multi broker:${NC} OFF (single broker = rank 0)"
fi

if [ "$ENABLE_REMOTE_GET_PREFETCH" == "1" ]; then
  echo -e "🚚 ${GREEN}Remote GET prefetch:${NC} ON"
  echo -e "  🔭 ${GREEN}  - LOOKAHEAD:${NC} ${REMOTE_GET_PREFETCH_LOOKAHEAD}"
  echo -e "  🧵 ${GREEN}  - MAX_INFLIGHT:${NC} ${REMOTE_GET_PREFETCH_MAX_INFLIGHT}"
  echo -e "  ⏱️  ${GREEN}  - WAIT_TIMEOUT:${NC} ${REMOTE_GET_PREFETCH_WAIT_TIMEOUT}"
else
  echo -e "🚚 ${YELLOW}Remote GET prefetch:${NC} OFF"
fi

echo -e "🗃️  ${GREEN}Custom Group Cache:${NC} LOOKAHEAD=${CUSTOM_GROUP_CACHE_LOOKAHEAD}, GC_INTERVAL=${CUSTOM_GROUP_GC_INTERVAL}, MAX_GROUPS=${CUSTOM_GROUP_CACHE_MAX_GROUPS}, MAX_LOCAL_PER_RANK=${CUSTOM_GROUP_CACHE_MAX_LOCAL_PER_RANK}"


echo -e ""
echo -e "${BLUE}------------------ CUDA / PyTorch config ------------------${NC}"

echo -e "🧩 ${GREEN}CUDA_VISIBLE_DEVICES:${NC} ${CUDA_VISIBLE_DEVICES}"
echo -e "🔗 ${GREEN}CUDA_DEVICE_MAX_CONNECTIONS:${NC} ${CUDA_DEVICE_MAX_CONNECTIONS}"

if [ "$NVTE_FLASH_ATTN" == "1" ]; then
  echo -e "⚡ ${GREEN}NVTE FlashAttention:${NC} ON"
else
  echo -e "⚡ ${YELLOW}NVTE FlashAttention:${NC} OFF"
fi

if [ "$NVTE_FUSED_ATTN" == "1" ]; then
  echo -e "🧬 ${GREEN}NVTE fused attention:${NC} ON"
else
  echo -e "🧬 ${YELLOW}NVTE fused attention:${NC} OFF"
fi

echo -e "🧠 ${GREEN}PyTorch CUDA Allocator:${NC} ${PYTORCH_CUDA_ALLOC_CONF}"


echo -e ""
echo -e "${BLUE}------------------ Data / debug ------------------${NC}"

if [ "$USE_FAKE_BATCH" == "1" ]; then
  echo -e "🧪 ${GREEN}Fake batch:${NC} ON (synthetic batches)"
else
  echo -e "🧪 ${YELLOW}Fake batch:${NC} OFF (real data path)"
fi

if [ -n "$DEBUG_DIR" ]; then
  echo -e "🐞 ${GREEN}Debug mode:${NC} ON"
  echo -e "  📁 Debug output dir: ${DEBUG_DIR}"
else
  echo -e "🐞 ${YELLOW}Debug mode:${NC} OFF"
fi


echo -e "${BLUE}==============================================${NC}"
echo "Config check done; launching..."
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
MERGE_FILE=<CKPT_ROOT>/gpt_2_merge.txt
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
    --train-iters ${TRAIN_ITERS:-1000}
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
