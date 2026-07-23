#!/bin/bash

OOM_RESTART_CODE=100
max_restarts=10
restart_count=0

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Error: Node Rank parameter is missing.${NC}"
    echo "Usage: $0 <node_rank>"
    exit 1
fi
NODE_RANK=$1

echo -e "${GREEN}Guardian script started for Node Rank: ${NODE_RANK}${NC}"

export ENABLE_PROFILING_DIT=0

# Main experiment selection (override on the command line, e.g.
#   RESOLUTION=1080p bash start_many.sh <node_rank>
#   RESOLUTION=360p SCHEDULE_TYPE=megatron-lm bash start_many.sh <node_rank>
# RESOLUTION picks the plan set + cost-model data; MAX_FRAMES is auto-derived in
# run_unified_many.sh (1080p -> 57, else -> 129). SCHEDULE_TYPE = genetic (ours) /
# megatron-lm / flex_sp.
export RESOLUTION="${RESOLUTION:-720p}"
export SCHEDULE_TYPE="${SCHEDULE_TYPE:-genetic}"

while true
do
    if [ "$ENABLE_PROFILING_DIT" == "1" ]; then
        echo -e "${BLUE}Profiling enabled. Running Node ${NODE_RANK} (${RESOLUTION}/${SCHEDULE_TYPE})... (Attempt: $((restart_count + 1)))${NC}"
        GBS=4 USE_FAKE_BATCH=1 bash run_unified_many.sh ${NODE_RANK} 1 2 49 1280 720
    else
        echo -e "${BLUE}Training Mode. Running Node ${NODE_RANK} (${RESOLUTION}/${SCHEDULE_TYPE})... (Attempt: $((restart_count + 1)))${NC}"
        GBS=8 USE_FAKE_BATCH=1 bash run_unified_many.sh ${NODE_RANK} 1 2 49 1280 720
    fi

    exit_code=$?
    echo -e "${BLUE}--- Task finished (Node ${NODE_RANK}) with exit code: ${exit_code} ---${NC}"

    if [ $exit_code -eq $OOM_RESTART_CODE ]; then
        ((restart_count++))
        if [ $restart_count -ge $max_restarts ]; then
            echo -e "${YELLOW}Warning (Node ${NODE_RANK}): Maximum restart limit (${max_restarts}) reached. Terminating.${NC}"
            break
        fi
        echo -e "${YELLOW}🚨 OOM signal detected (Code 100). Restarting in 5 seconds...${NC}"
        sleep 5
        continue 
        
    elif [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Command completed successfully (Node ${NODE_RANK}). Exiting guardian loop.${NC}"
        break
        
    else
        echo -e "${RED}An unknown error occurred (Node ${NODE_RANK}, exit code ${exit_code}). Not OOM. Terminating.${NC}"
        break
    fi
done

echo ""
echo "Automation script has finished (Node ${NODE_RANK})."
