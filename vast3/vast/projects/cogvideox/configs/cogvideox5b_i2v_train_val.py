import os

# 数据w h t
dst_size = (720, 480)
num_frames = 25
# 训练配置
config = dict(
    ## log&ckpts路径
    project_dir=os.path.join(os.getcwd(), "work_dirs/cogvideox/cogvideox_i2v20241223"),
    runners=["projects.cogvideox.adaptors.CogVideoXTrainer"],
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        # gpu_ids=[6, 7],
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
        train=dict(
            dataset=dict(
                type="ConcatDataset",
                data_path_list=[
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/7w/new_pack/pack_66/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/7w/new_pack/pack_67/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_1/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_2/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_3/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_4/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_5/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_6/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_7/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_8/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_9/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_10/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_11/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_6/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_7/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_8/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_9/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_10/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_11/v0.0.1",
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=1,
            filter=dict(
                mode="overall_func",
                func="projects.cogvideox.configs.cogvideox_i2v.filter_data",
                dst_size=dst_size,
                min_num_frames=100,
                multiple=16,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type="CogVideoXTransform",
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                mask_ratios={
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
                is_train=True,
                prompt_cfg=dict(
                    model_name="t5",
                    model_path="/root/model_zoo/pretrained/models--THUDM--CogVideoX-5b",
                    prompt_mode="from_dict",
                    prompt_embeds_mode="default",  # from dict
                    default_prompt_prob=0.2,
                    clean_prompt=True,
                    max_length=226,
                    with_attention_mask=True,
                    prompt_names=["prompt_cogvlm2", "prompt_panda"],
                ),
            ),
            sampler=dict(
                type="BucketSampler",
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        #### dataloader eval配置
        eval=dict(
            dataset=dict(
                type="ConcatDataset",
                data_path_list=[
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_zwzx_2_sample/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_zwzx_3_sample/v0.0.1",
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=1,
            filter=dict(
                mode="overall_func",
                func="projects.cogvideox.configs.cogvideox_i2v.filter_data",
                dst_size=dst_size,
                min_num_frames=100,
                multiple=16,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type="CogVideoXTransform",
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                is_train=True,
                prompt_cfg=dict(
                    model_name="t5",
                    model_path="/root/model_zoo/pretrained/models--THUDM--CogVideoX-5b",
                    prompt_mode="from_dict",
                    default_prompt_prob=0.0,
                    clean_prompt=True,
                    max_length=226,
                    with_attention_mask=True,
                    prompt_names=["prompt_cogvlm2", "prompt_panda"],
                ),
            ),
            sampler=dict(
                type="DefaultSampler",
                infinite=False,
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
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
        eval_interval=10,
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
