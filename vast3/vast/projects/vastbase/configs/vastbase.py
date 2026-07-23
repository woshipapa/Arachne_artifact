import os

# 数据w h t
dst_size = (1280, 736)
num_frames = 49

# Temporary code for quick debugging
debug = False
if debug:
    GPU_IDS = [0]
    NUM_WORKERS = 0
    import logging
    logging.basicConfig(level=logging.DEBUG)
else:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    # GPU_IDS = [0, 1]
    NUM_WORKERS = 2 #不能设置太大，transform中需要loadT5,设置太大，load次数太多

# 训练配置
config = dict(
    ## log&ckpts路径
    project_dir=os.path.join(os.getcwd(),'projects/vastbase/experiments_sjl/vast/vast_20250110_pre'),
    runners=["projects.vastbase.adaptors.VASTBASETrainer"],    
    ## 分布式配置for luancher
    launch=dict(
        gpu_ids=GPU_IDS,
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file=os.path.join(
                os.getcwd(),'configs/accelerate_configs/zero2.json'
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
                type="ConcatDataset",
                data_path_list=[
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_20_1_1/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_20_1_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_20_2_new/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_3/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_4/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_5/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_6/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_36_7/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_1_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_1_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_2_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_2_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_3_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_3_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_4_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_4_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_5_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_5_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_6_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_6_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_55_7/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_66_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_66_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_67_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_67_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_67_3/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_67_4/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_71_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_71_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_zwzx_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_zwzx_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack/pack_zwzx_3/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_3_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_4_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_5_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_6_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_36_7_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_1_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_1_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_2_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_2_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_3_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_3_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_4_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_4_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_5_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_5_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_6_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_6_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_55_7_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_66_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_66_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_67_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_67_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_67_3_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_67_4_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_71_1_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_71_2_slice_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_pack_20_1_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_pack_20_1_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_pack_20_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v1/200w/new_pack_no_body/pack_zwzx_1_slice_new_0/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/7w/new_pack/pack_66/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/7w/new_pack/pack_67/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_1/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_2/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_3/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_4/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_5/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_6/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_7/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_8/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_9/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_10/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_11/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_6/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_7/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_8/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_9/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_10/v0.0.1",
                    # "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_11/v0.0.1",
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            filter=dict(
                mode='overall_func',
                func='projects.vastbase.configs.vastbase.filter_data',
                dst_size=dst_size,
                min_num_frames=num_frames * 4,
                multiple=32,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type='VASTBASETransform',
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                mask_ratios={
                    'image_head': 0.15,
                    'image_tail': 0.05,
                    'image_head_tail': 0.1,
                    'quarter_head': 0.05,
                    'quarter_tail': 0.03,
                    'quarter_head_tail': 0.03,
                    'quarter_random': 0.03,
                    'interpolation': 0.03,
                    'random': 0.03,
                },  # 50%
                is_train=True,
                prompt_cfg=dict(
                    model_name='t5',
                    model_path=os.path.join(os.getcwd(),'pretrained/models--THUDM--CogVideoX-5b'),
                    prompt_mode='from_dict',
                    default_prompt_prob=0.2,
                    clean_prompt=True,
                    max_length=226,
                    with_attention_mask=True,
                    prompt_names=["prompt_cogvlm2", "prompt_panda"],
                ),
            ),
            sampler=dict(
                type='BucketSampler',
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
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_11/v0.0.1",
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=NUM_WORKERS,
            filter=dict(
                mode='overall_func',
                func='projects.vastbase.configs.vastbase.filter_data',
                dst_size=dst_size,
                min_num_frames=100,
                multiple=32,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type='VASTBASETransform',
                size_method=2,
                dst_size=dst_size,
                flip=False,
                num_frames=num_frames,
                stride=4,
                cn_mode=None,
                mask_ratios={
                    'image_head': 0.15,
                    'image_tail': 0.05,
                    'image_head_tail': 0.1,
                    'quarter_head': 0.05,
                    'quarter_tail': 0.03,
                    'quarter_head_tail': 0.03,
                    'quarter_random': 0.03,
                    'interpolation': 0.03,
                    'random': 0.03,
                },  # 50%
                is_train=True,
                prompt_cfg=dict(
                    model_name='t5',
                    model_path=os.path.join(os.getcwd(),'pretrained/models--THUDM--CogVideoX-5b'),
                    prompt_mode='from_dict',
                    default_prompt_prob=0.2,
                    clean_prompt=True,
                    max_length=226,
                    with_attention_mask=True,
                    prompt_names=["prompt_cogvlm2", "prompt_panda"],
                ),
            ),
            sampler=dict(
                type='BucketSampler',
                infinite=False,
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
    ),
    ### 模型model配置
    models=dict(
        pretrained=os.path.join(os.getcwd(),'pretrained/models--THUDM--CogVideoX-5b'),
        transformer_pretrained=os.path.join(os.getcwd(),'pretrained/cxz_720p_v2'),
        loss=dict(),
        from_giga=True,
        trainable_params = ["patch_embed.proj.1", "proj_out_lr", "norm_final_lr", "norm_out_lr"],
        # trainable_params = ["patch_embed.proj.1", "proj_out_lr","transformer_blocks_front","transformer_blocks_midds","norm_final_lr","norm_out_lr"],
        # trainable_params = ["guider", "patch_embed.proj", "transformer.transformer_blocks_midss", "transformer.transformer_blocks_midms"],
        patch_size=[(1,2,2),(1,4,4)],
        only_orgin=True
    ),
    ### 优化器optimizer配置
    optimizers=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.995),  # β1和β2
        eps=1e-8,            # 防止除以零的常数
    ),
    ### 学习率scheduler配置
    schedulers=dict(
        # type='ConstantScheduler',
        type='CosineScheduler',
    ),
    ### 训练过程train配置
    train=dict(
        resume=True,
        checkpoint_save_optimizer=True,
        max_epochs=20,
        gradient_accumulation_steps=8,
        mixed_precision='bf16',  # fp16, bf16
        checkpoint_interval=50,
        eval_interval=10,
        checkpoint_total_limit=-1,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['VASTBlockDS', 'VASTBlockSS'],
    ),
    ### 测试过程test配置
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
            # import pdb; pdb.set_trace()
            video_length = data_dict['video_length']
            video_height = data_dict['video_height']
            video_width = data_dict['video_width']
            if video_height * video_width < min_area:
                continue
            if 'scores' in data_dict:
                optical_flow_unimatch = data_dict['scores']['optical_flow_unimatch']
                aesthetic = data_dict['scores']['aesthetic']
                if optical_flow_unimatch < 2:
                    continue
                if aesthetic < 4:
                    continue
            if video_length < min_num_frames:
                continue
            # image size
            dst_width, dst_height = image_utils.get_image_size(
                (video_width, video_height),
                dst_size,
                mode='area',
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
                data_dict['bucket_index'] = bucket_index
                data_dict['video_info'] = video_info
                # data_dict['prompt'] = data_dict['text']
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list

