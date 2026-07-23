#!/bin/bash
# =============================================================================
#
#
# Purpose: verify on a SINGLE machine that the training pipeline actually
#          enters the training loop and the environment is healthy. No
#          multi-node rendezvous, no real data (fake batch), only a few
#          iterations. Seeing per-iteration training logs == env + code path OK.
#
#     bash run_single_node_test.sh
#
#
#               NUM_LAYERS=2 NUM_SINGLE_LAYERS=2 bash run_single_node_test.sh
#
#           python check_env.py
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-12344}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NGPU=0
    fi
    if [ "$NGPU" -ge 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ "$NGPU" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        echo "[ERROR] No GPU detected. This smoke test needs at least one GPU to enter training."
        echo "   No GPU detected. This smoke test needs at least 1 GPU."
        echo "   To check dependencies only (no GPU required), run:  python check_env.py"
        exit 2
    fi
fi

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
TP=${TP:-1}
if [ "$NGPU" -ge 2 ]; then
    CP=${CP:-2}
else
    CP=${CP:-1}
fi

if [ $(( NGPU % (TP * CP) )) -ne 0 ]; then
    echo "[ERROR] Invalid configuration: the visible GPU count ($NGPU) is not divisible by TP*CP ($((TP*CP)))."
    echo "   Adjust CUDA_VISIBLE_DEVICES / TP / CP."
    exit 3
fi

export TRAIN_ITERS=${TRAIN_ITERS:-3}
export GBS=${GBS:-2}
export USE_FAKE_BATCH=1

echo "===================== smoke test config ====================="
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES  (GPUs = $NGPU)"
echo "  TP = $TP   CP = $CP   DP = $(( NGPU / (TP*CP) ))"
echo "  TRAIN_ITERS = $TRAIN_ITERS   GBS = $GBS   USE_FAKE_BATCH = 1"
echo "  model/resolution = hunyuan / 49 frames / 1280x720"
echo "  MASTER = $MASTER_ADDR:$MASTER_PORT  (WORLD_SIZE=NNODES=1)"
echo "======================================================================="
echo ""

bash run_unified_many.sh 0 "$TP" "$CP" 49 1280 720
rc=$?

echo ""
if [ $rc -eq 0 ]; then
    echo "[OK] Smoke test passed (exit 0): entered the training loop and completed ${TRAIN_ITERS} steps. Environment is ready."
    echo "   Smoke test passed: reached the training loop and ran ${TRAIN_ITERS} iterations."
else
    echo "[ERROR] Smoke test failed (exit $rc): check the log above to locate the problem."
    echo "   Smoke test failed (exit $rc). See the log above."
fi
exit $rc
