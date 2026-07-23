import os

# --- Env-driven experiment selection (same convention as the main branch) ---
_RES_MAP = {"360p": (656, 352), "720p": (1280, 720), "1080p": (1936, 1072)}
RESOLUTION = os.environ.get("RESOLUTION", "720p")
dst_size = _RES_MAP.get(RESOLUTION, (1280, 720))
dst_num_frames = int(os.environ.get("MAX_FRAMES") or ("57" if RESOLUTION == "1080p" else "129"))
dst_fps = 15

_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
def _pick_sim_log(model):
    if os.environ.get("SIM_LOG"):
        return os.environ["SIM_LOG"]
    cands = []
    if os.environ.get("MAX_FRAMES"):
        cands.append(os.path.join(
            _REPO_ROOT, f"simulation_log_{model}_{RESOLUTION}_{os.environ['MAX_FRAMES']}.txt"))
    cands.append(os.path.join(_REPO_ROOT, f"simulation_log_{model}_{RESOLUTION}.txt"))
    for _c in cands:
        if os.path.exists(_c):
            return _c
    return cands[-1]
_SIM_LOG = _pick_sim_log("cogvideox")
# 训练配置
config = dict(
    ## log&ckpts路径
    # project_dir=os.path.join(os.getcwd(), "work_dirs/cogvideox/cogvideox_i2v20241223"),
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
        num_machines=os.environ.get("WORLD_SIZE", 1),
        until_completion=True,
        main_process_ip=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        main_process_port=int(os.environ.get("MASTER_PORT", "12346"))
    ),
    ## 训练配置for runner
    ### dataloader配置
    dataloaders=dict(
        #### dataloader train配置
        train=dict(
            dataset=dict(
                # type="ConcatDataset",
                type="ExpDataset",
                log_file_path=_SIM_LOG,
                ranks_to_process=[0,1,2,3,4,5,6,7],
                data_path_list=[
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
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_6/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_7/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_8/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_9/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_10/v0.0.1",
                    "/root/datasets/giga_datasets/train_data/cogvideo_5b_v0.0.3/50w/anno_pack/part_other_11/v0.0.1",
                ],
                transforms=[
                    dict(
                        type="SampleImages",
                        num_frames=dst_num_frames,
                    ),
                    dict(
                        type="PromptGenerator",
                        clean_prompt=True,
                        short_prompt_prob=0,
                        # default_prompt_prob=0.1,
                        default_prompt_prob=0,
                    ),
                    dict(
                        type="PromptToTransformerEmbedding",
                        model_name="llama",
                        # model_path="/data01/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder",
                        model_path="/root/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder",
                        max_length=256,
                        with_attention_mask=True,
                    ),
                    dict(
                        type="PromptToClipEmbedding",
                        # model_path="/data01/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder_2",
                        model_path="/root/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder_2",
                    ),
                    dict(
                        type="GenerateFirstRefImage",
                    ),
                    dict(
                        type="PackInputs",
                        image_keys=["images", "first_ref_image"],
                        embedding_keys=[
                            "prompt_embeds",
                            "prompt_masks",
                            "clip_text_embed",
                        ],
                        dst_size=dst_size,
                    ),
                ],
            ),
            batch_size_per_gpu=1,
            num_workers=0,
            filter=dict(
                mode="overall_func",
                func="projects.cogvideox.configs.cogvideox_i2v.filter_data",
                dst_size=dst_size,
                min_num_frames=100,
                multiple=16,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            # transform=dict(
            #     type="CogVideoXTransform",
            #     size_method=2,
            #     dst_size=dst_size,
            #     flip=False,
            #     num_frames=num_frames,
            #     stride=4,
            #     cn_mode=None,
            #     mask_ratios={
            #         "image_head": 0.15,
            #         "image_tail": 0.05,
            #         "image_head_tail": 0.1,
            #         "quarter_head": 0.05,
            #         "quarter_tail": 0.03,
            #         "quarter_head_tail": 0.03,
            #         "quarter_random": 0.03,
            #         "interpolation": 0.03,
            #         "random": 0.03,
            #     },  # 50%
            #     is_train=True,
            #     prompt_cfg=dict(
            #         model_name="t5",
            #         model_path="/root/model_zoo/pretrained/models--THUDM--CogVideoX-5b",
            #         prompt_mode="from_dict",
            #         default_prompt_prob=0.2,
            #         clean_prompt=True,
            #         max_length=226,
            #         with_attention_mask=True,
            #         prompt_names=["prompt_cogvlm2", "prompt_panda"],
            #     ),
            # ),
            sampler=dict(
                # type="BucketSampler",
                type="DefaultSampler"
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
            in_channels=16,  # with ref images 16->32, with ref and cn_images 16->48
        ),
        loss=dict(),
        vae=dict(
            vae_slicing=False,
            vae_tiling=True,
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
                if aesthetic < 4:
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
                # data_dict['prompt'] = data_dict['text']
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list
