import os

from vast.datasets.config.t2v_200w import get_data_list

# 数据w h t
dst_size = (720, 480)
num_frames = 25
# 训练配置
config = dict(
    ## log&ckpts路径
    runners=["projects.cogvideox.adaptors.CogVideoXTrainer"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(), "configs/accelerate_configs/zero2.json"
            ),
        ),
        until_completion=True,
    ),
    ## 训练配置for runner
    ### dataloader配置
    dataloaders=dict(
        #### dataloader train配置
        train=dict(
            dataset=dict(
                type="ClipDataset",
                data_path_list=get_data_list(),
                transforms=[
                    dict(
                        type="SampleImages",
                        num_frames=num_frames,
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
                        model_name="t5",
                        model_path="/root/model_zoo/pretrained/models--THUDM--CogVideoX-5b",
                        max_length=226,
                        with_attention_mask=True,
                    ),
                    dict(
                        type="GenerateRefImages",
                        mask_cfg={
                            "image_head": 0.15,
                            "image_tail": 0.05,
                            "image_head_tail": 0.1,
                            "quarter_head": 0.05,
                            "quarter_tail": 0.03,
                            "quarter_head_tail": 0.03,
                            "quarter_random": 0.03,
                            "interpolation": 0.03,
                            "random": 0.03,
                        },  # 50%
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=["images", "ref_images"],
                        embedding_keys=["prompt_embeds", "prompt_masks"],
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
        pretrained="/root/model_zoo/pretrained/models--THUDM--CogVideoX-5b",
        transformer_pretrained="/root/model_zoo/pretrained/transformer_giga",
        transformer=dict(
            in_channels=32,  # with ref images 16->32, with ref and cn_images 16->48
        ),
        loss=dict(),
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
        resume=False,
        checkpoint_save_optimizer=True,
        max_epochs=10,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",  # fp16, bf16
        checkpoint_interval=3000,
        checkpoint_total_limit=-1,
        log_with="tensorboard",
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=["CogVideoXBlock"],
    ),
    ### 测试过程test配置
    test=dict(),
)
