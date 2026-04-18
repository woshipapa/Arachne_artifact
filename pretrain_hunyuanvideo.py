# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import json
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.hunyuan.pipeline import HunyuanPipeline
from megatron.core.models.hunyuan import DiTPipeline,VAEPipeline
from megatron.core.models.hunyuan.WrapedModel import TrainingWrapperModel
from megatron.core import mpu
from hunyuanvideo_dataset_builder import HunyuanVideoDatasetBuilder
from megatron.core.datasets.hunyuanvideo_dataset_config import HunyuanVideoDatasetConfig

from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0,get_model
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.training import pretrain
from vast.train.configs.config import load_config
from hunyuanvideo.configs.hunyuanvideo_i2vhy import config
from megatron.training.global_vars import set_global_config
from megatron.training import get_global_config
from megatron.training.utils import (
    get_batch_on_this_tp_cp_rank_vast,
    get_batch_on_this_tp_rank_vast,
    average_losses_across_data_parallel_group
)
import torch.distributed as dist
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
from megatron.training import build_train_valid_test_data_iterators
import numpy as np
import random





class Config(dict):
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def get_batch(data_iterator):
    batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
    return batch


def extra_args_provider(parser):
    group = parser.add_argument_group(title='dataset')
    group.add_argument("--num-frames", type=int)
    group.add_argument("--video-resolution", nargs=2, type=int)
    group = parser.add_argument_group(title='log')
    group.add_argument("--debug", action="store_true", help="debug mode")
    group.add_argument("--debug-dir", type=str, default="debug", help="debug dir")
    group.add_argument("--global-log-dir", type=str, help="global log dir")
    group.add_argument("--handler-log-dir", type=str, help="handler log dir")
    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    from vast.datasets.datasets.build import build_dataset as build_dataset_vast
    from megatron.core.datasets.fake.build import build_dataset

    global_config = get_global_config()
    train_ds_config = global_config.dataloaders.train
    eval_ds_config = global_config.dataloaders.eval

    ds_config = HunyuanVideoDatasetConfig(
        train_ds_config=train_ds_config,
        eval_ds_config=eval_ds_config
    )

    print_rank_0("> building train, validation, and test datasets for multimodal ...")
    
    if "FakeDataset" == train_ds_config.dataset.type:
        train_ds = build_dataset(train_ds_config.dataset)
        valid_ds = None
        test_ds = None
    else:
        dataset = build_dataset_vast(train_ds_config.dataset)
        train_ds, valid_ds, test_ds = HunyuanVideoDatasetBuilder(
            dataset,
            train_val_test_num_samples,
            lambda: True,
            ds_config,
        ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds

def init(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
):
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )

    args = get_args()
    

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> HunyuanPipeline:
    args = get_args()
    global_config = get_global_config()
    hunyuan_config=global_config.models

    config = core_transformer_config_from_args(args)

    

    model_str = os.environ.get("MODEL_TYPE")
    if model_str == "wan":
        from megatron.core.models.wan.WrapedModel import TrainingWrapperModel
        model = TrainingWrapperModel(wan_config = hunyuan_config, dit_model_config = config)
    elif model_str == "hunyuan":
        from megatron.core.models.hunyuan.WrapedModel import TrainingWrapperModel
        model = TrainingWrapperModel(hunyuan_config=hunyuan_config,dit_model_config=config)
    elif model_str == "cogvideox":
        from megatron.core.models.cogvideox.WrapedModel import TrainingWrapperModel
        model = TrainingWrapperModel(cog_config=hunyuan_config, dit_model_config=config)
    
    
    
    return model





if __name__ == "__main__":
    global_config = load_config(config)
    set_global_config(global_config)
    from t2v_flow.executor import DynamicForwardStepHandler
    handler = DynamicForwardStepHandler()

    

    
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        handler.forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
