#!/bin/bash
export ENABLE_PROFILING_DIT=0
CMD="GBS=8 USE_FAKE_BATCH=1 bash run_unified.sh 1 2 720 1280 49"


if [ "$ENABLE_PROFILING_DIT" == "1" ]; then
    echo "Profiling enabled. Running with nsys..."
    GBS=4 USE_FAKE_BATCH=1 bash run_unified.sh 1 2 720 1280 49
else
    echo "Profiling disabled. Running original command..."
    GBS=4 USE_FAKE_BATCH=1 bash run_unified.sh 1 1 49 1280 720
fi