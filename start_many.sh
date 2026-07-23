#!/bin/bash
# DeepSpeed-baseline unified entry point — same convention as the main branch:
#
#   MODEL_TYPE=hunyuan   RESOLUTION=720p  bash start_many.sh <node_rank>
#   MODEL_TYPE=cogvideox RESOLUTION=720p  bash start_many.sh <node_rank>
#   MODEL_TYPE=wan1.3b   RESOLUTION=720p  bash start_many.sh <node_rank>
#
# Env vars (all optional except <node_rank>):
#   MODEL_TYPE  : hunyuan (default) | cogvideox | wan / wan1.3b
#   RESOLUTION  : 360p | 720p (default) | 1080p    -> dst_size in the config
#   MAX_FRAMES  : frame window; auto-derived (1080p -> 57, else 129)
#   WORLD_SIZE  : number of NODES (default 2, 8 GPUs each).
#                 WORLD_SIZE=4 / 8 automatically selects the node-scaling
#                 workload (hunyuan; replays all 8 samples/iter), same
#                 convention as the main branch
#   MASTER_ADDR : node-0 IP (default 127.0.0.1); MASTER_PORT (default 12346)
#   SIM_LOG     : explicit workload-replay log override; defaults to the
#                 repo-root simulation_log_<model>_<RESOLUTION>[...].txt

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Error: Node Rank parameter is missing.${NC}"
    echo "Usage: $0 <node_rank>"
    exit 1
fi
export RANK=$1

export MODEL_TYPE="${MODEL_TYPE:-hunyuan}"
export RESOLUTION="${RESOLUTION:-720p}"
export WORLD_SIZE="${WORLD_SIZE:-2}"

case "$MODEL_TYPE" in
  hunyuan)      CONFIG=projects/hunyuanvideo/configs/hunyuanvideo_i2vhy.py ;;
  cogvideox)    CONFIG=projects/cogvideox/configs/cogvideox_i2v.py ;;
  wan|wan1.3b)  CONFIG=projects/wan/configs/wan_i2vhy.py ;;
  *) echo -e "${RED}Unknown MODEL_TYPE: $MODEL_TYPE (hunyuan|cogvideox|wan1.3b)${NC}"; exit 1 ;;
esac

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Reuse the Arachne artifact's .venv and apply the baseline-specific setup
# (VAE patch + my_utils). No-op once both markers are present.
bash "$ROOT/setup_env.sh" || {
    echo -e "${RED}Error: environment setup failed; aborting before launch.${NC}"
    exit 1
}
source "$ROOT/.venv/bin/activate"

# Four entries, all required:
#   accelerate/src    -> the customized accelerate (NOT the pip-installed one;
#                        must precede it on the path)
#   vast3/vast        -> resolves `import vast` (package is at vast3/vast/vast/)
#   vast3/vast/my_utils -> nested my_utils package
#   $ROOT             -> teleai_data_tool (vendored, shared with the main branch)
export PYTHONPATH=$ROOT/vast3/accelerate/src:$ROOT/vast3/vast:$ROOT/vast3/vast/my_utils:$ROOT:$PYTHONPATH

cd "$ROOT/vast3/vast" || exit 1
echo -e "${GREEN}[DeepSpeed baseline] MODEL_TYPE=$MODEL_TYPE RESOLUTION=$RESOLUTION" \
        "MAX_FRAMES=${MAX_FRAMES:-auto} RANK=$RANK WORLD_SIZE=$WORLD_SIZE${NC}"
echo -e "${GREEN}[DeepSpeed baseline] CONFIG=$CONFIG${NC}"

python tools/train.py "$CONFIG"
