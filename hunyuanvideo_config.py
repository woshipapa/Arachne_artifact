import os

dst_size = (720, 480)
num_frames = 25

# Temporary code for quick debugging
debug = False
if debug:
    GPU_IDS = [0]
    NUM_WORKERS = 0
    import logging

    logging.basicConfig(level=logging.DEBUG)
else:
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    NUM_WORKERS = 2

config = dict(
    # 改到存储空间足够的地址
    project_dir=os.path.join(os.getcwd(), "work_dirs/hunyuanvideo/hunyuanvideo_t2v"),
    runners=["projects.hunyuanvideo.adaptors.HunYuanVideoTrainer"],
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
    dataloaders=dict(
        train=dict(
            dataset=dict(
                type="ConcatDataset",
                data_path_list=[
                    "/root/v0.0.1",
                    # "/root/v0.0.1"/
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            filter=dict(
                mode="overall_func",
                func=".hunyuanvideo.configs.hunyuanvideo.filter_data",
                dst_size=dst_size,
                min_num_frames=num_frames * 4,
                multiple=16,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type="HunYuanVideoTransform",
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                with_ref_images=True,  # i2v or t2v
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
                    dtype="bfloat16",
                    model_name="llama",
                    model_path="/root/text_encoder",
                    clip_model_path="/root/text_encoder_2",
                    prompt_mode="from_dict",
                    prompt_embeds_mode="default",
                    default_prompt_prob=0.2,
                    clean_prompt=True,
                    max_length=226,  # 可以适当调节
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
        #### validation
        eval=dict(
            dataset=dict(
                type="ConcatDataset",
                data_path_list=[
                    "/root/v0.0.1",
                    "/root/v0.0.1",
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=1,
            filter=dict(
                mode="overall_func",
                func="projects.hunyuanvideo.configs.hunyuanvideo.filter_data",
                dst_size=dst_size,
                min_num_frames=50,
                multiple=16,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type="HunYuanVideoTransform",
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                with_ref_images=True,  # i2v or t2v
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
                    dtype="bfloat16",
                    model_name="llama",
                    model_path="/root/text_encoder",
                    clip_model_path="/root/text_encoder_2",
                    prompt_mode="from_dict",
                    prompt_embeds_mode="default",
                    default_prompt_prob=0.2,
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
    models=dict(
        pretrained="/root/hunyuanvideo_13b",
        transformer=dict(
            in_channels=32,  # t2v 16 or i2v 32
        ),
        # vae config
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
        checkpoint_interval=500,
        eval_interval=100,
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
    test=dict(),
)


# 数据在线筛选和过滤
def filter_data(
    all_data_list,
    dst_size,
    min_num_frames,
    max_num_frames=None,
    multiple=16,
    min_area=-1,
    min_size=1,
):
    from vast.datasets import image_utils

    video_info_dict = dict()
    for n, data_list in enumerate(all_data_list):
        for m, data_dict in enumerate(data_list):
            video_height = data_dict["video_height"]
            video_width = data_dict["video_width"]
            if video_height * video_width < min_area:
                continue
            if "scores" in data_dict:
                optical_flow_unimatch = data_dict["scores"]["optical_flow_unimatch"]
                aesthetic = data_dict["scores"]["aesthetic"]
                if optical_flow_unimatch < 2:
                    continue
                if aesthetic < 5:
                    continue

            video_valid_range = data_dict.get("video_valid_range", None)
            if video_valid_range is None:
                video_valid_range = (0, data_dict.get("video_length", 0))
            video_valid_range = list(video_valid_range)
            video_length = video_valid_range[1] - video_valid_range[0]
            if video_length < min_num_frames:
                continue
            data_dict["video_valid_range"] = video_valid_range
            data_dict["video_length"] = video_length
            # image size
            dst_width, dst_height = image_utils.get_image_size(
                (video_width, video_height),
                dst_size,
                mode="area",
                multiple=multiple,
            )
            video_info = (dst_width, dst_height)
            if video_info not in video_info_dict:
                video_info_dict[video_info] = []
            video_info_dict[video_info].append((n, m))

    new_all_data_list = [[] for _ in range(len(all_data_list))]
    bucket_index = 0
    for video_info, data_indexes in video_info_dict.items():
        if len(data_indexes) >= min_size:
            for n, m in data_indexes:
                data_dict = all_data_list[n][m]
                data_dict["bucket_index"] = bucket_index
                data_dict["video_info"] = video_info
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list


def get_pose_len(num_poses):
    from collections import Counter

    count_poses = Counter(num_poses)
    count_poses = count_poses.most_common(2)
    if len(count_poses) == 1:
        pose_len = count_poses[0][0]
    else:
        keys = [count_poses[0][0], count_poses[1][0]]
        values = [count_poses[0][1], count_poses[1][1]]
        if values[0] == values[1]:
            pose_len = max(keys)
        elif values[0] > values[1]:
            pose_len = keys[0]
        else:
            pose_len = keys[1]
    return pose_len
