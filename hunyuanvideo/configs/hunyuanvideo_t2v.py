import os

from vast.datasets.config.t2v_200w import get_data_list

# 数据w h t
dst_size = (1280, 720)
num_frames = 25
debug = False
# debug = True
if debug:
    GPU_IDS = [0]
    NUM_WORKERS = 0
    import logging

    logging.basicConfig(level=logging.DEBUG)
else:
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    NUM_WORKERS = 2
# 训练配置
config = dict(
    ## log&ckpts路径
    runners=["projects.hunyuanvideo.adaptors.HunYuanVideoTrainer"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=GPU_IDS,
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=True,
    ),
    ## 训练配置for runner
    ### dataloader配置
    dataloaders=dict(
        #### dataloader train配置
        train=dict(
            dataset=dict(
                type="ClipDataset",
                data_path_list=get_data_list(video_type_list=['other']),
                transforms=[
                    dict(
                        type="SampleImages",
                        num_frames=num_frames*4,
                        stride=4,
                    ),
                    dict(
                        type="PromptGenerator",
                        clean_prompt=True,
                        short_prompt_prob=0.5,
                        default_prompt_prob=0.2,
                    ),
                    dict(
                        type="PromptToTransformerEmbedding",
                        model_name="llama",
                        model_path="/root/text_encoder",
                        max_length=226,
                        with_attention_mask=True,
                    ),
                    dict(
                        type="PromptToClipEmbedding",
                        model_path="/root/text_encoder_2",
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=["images"],
                        embedding_keys=["prompt_embeds", "prompt_masks", "clip_text_embed"],
                        dst_size=dst_size,
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=2,
            sampler=dict(
                type="BucketSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        #### dataloader eval配置
        eval=None,
    ),
    ### 模型model配置
    models=dict(
        pretrained="/root/hunyuanvideo_13b",
        transformer_pretrained="/root/transformer",
        transformer=dict(
            in_channels=16,  # with ref images 16->32, with ref and cn_images 16->48
        ),
        loss=dict(),
        vae=dict(
            vae_slicing=True,
            vae_tiling=True,
        ),
        # flow matching schdule
        scheduler=dict(
            flow_resolution_shifting=False,
            flow_base_image_seq_len=256,
            flow_max_image_seq_len=4096,
            flow_base_shift=0.5,
            flow_max_shift=1.15,
            flow_shift=1.0,
            flow_weighting_scheme="none",
            flow_logit_mean=0.0,
            flow_logit_std=1.0,
            flow_mode_scale=1.29,
        ),
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type="AdamW",
        lr=2e-5,
        weight_decay=1e-2,
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        type="ConstantScheduler",
    ),
    ### 训练过程train配置
    train=dict(
        resume=True,
        checkpoint_save_optimizer=True,
        max_epochs=10,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",  # fp16, bf16
        checkpoint_interval=100,
        checkpoint_total_limit=-1,
        log_with="tensorboard",
        log_interval=1,
        with_ema=False,
        activation_checkpointing=True,
        activation_class_names=[
            "HunyuanVideoTransformerBlock",
            "HunyuanVideoSingleTransformerBlock",
        ],
    ),
    ### 测试过程test配置
    test=dict(),
)
