# SOURCE_CKPT_PATH="<CKPT_ROOT>/transformer"
SOURCE_CKPT_PATH="<CKPT_ROOT>/transformer_safetensor"
TARGET_CKPT_PATH="<CKPT_ROOT>/ckpt_tp1_2040_linearparallel_epoch1step2700"
TP=1
PP=1

python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --convert-checkpoint-from-megatron-to-transformers