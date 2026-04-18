from dataclasses import dataclass

import torch
from megatron.core.transformer.utils import openai_gelu

from megatron.core.models.hunyuan.model import HunyuanParams
from megatron.core.models.vae.autoencoder import AutoEncoderParams
from megatron.training import get_global_config

import os
@dataclass
class HunyuanModelParams:
    flux_params: HunyuanParams
    vae_params: AutoEncoderParams
    clip_params: dict | None
    t5_params: dict | None
    scheduler_params: dict | None
    device: str | torch.device


global_config = get_global_config()

# configs = {
#     "dev": FluxModelParams(
#         flux_params=FluxParams(
#             num_joint_layers=19,
#             num_single_layers=38,
#             hidden_size=3072,
#             num_attention_heads=24,
#             activation_func=openai_gelu,
#             add_qkv_bias=True,
#             ffn_hidden_size=16384,
#             in_channels=64,
#             context_dim=4096,
#             model_channels=256,
#             patch_size=1,
#             guidance_embed=True,
#             vec_in_dim=768,
#         ),
#         vae_params=AutoEncoderParams(
#             ch_mult=[1, 2, 4, 4],
#             attn_resolutions=[],
#             resolution=256,
#             in_channels=3,
#             ch=128,
#             out_ch=3,
#             num_res_blocks=2,
#             z_channels=16,
#             scale_factor=0.3611,
#             shift_factor=0.1159,
#             ckpt=None,
#         ),
#         clip_params={
#             'max_length': 77,
#             'always_return_pooled': True,
#         },
#         t5_params={
#             'max_length': 512,
#         },
#         scheduler_params={
#             'num_train_timesteps': 1000,
#         },
#         device='cpu',
#     )
# }

def get_root_dir():
    return os.path.abspath(__file__).split("vast")[0][:-1]


def get_model_dir():
    return os.environ.get("VAST_MODELS_DIR", "./models/")


def get_huggingface_model_path(model_name):
    # This conditional statement is for adapting to the low versions of Hugging Face,
    # HUGGINGFACE_HUB_CACHE variable will be deleted in later versions.
    model_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not model_dir:
        model_dir = os.environ.get("HF_HUB_CACHE")

    model_name = os.path.expandvars(model_name)
    if "/" in model_name and len(model_name.split("/")) == 2:
        local_model_name = "models--" + model_name.replace("/", "--")
        hf_model_name = model_name
    elif "--" in model_name and model_name.startswith("models"):
        local_model_name = model_name
        hf_model_name = model_name[8:]
        hf_model_name = hf_model_name.replace("--", "/")
    else:
        local_model_name = model_name
        hf_model_name = None
    model_path = os.path.join(model_dir, local_model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} does not exist")


def get_model_path(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f"{model_name_or_path} does not exist")
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)



