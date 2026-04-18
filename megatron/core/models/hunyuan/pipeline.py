# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Optional, Union, Any, Mapping

import functools
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from tqdm import tqdm
import logging
# from megatron.core.export.data_type import DataType
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from megatron.core import mpu

from megatron.core.models.hunyuan.model import HunyuanVideoTransformer3DModel, HunyuanParams
from megatron.core.models.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from megatron.core.models.hunyuan.hunyuan_pipeline_utils import HunyuanModelParams, get_model_path
from megatron.core.models.vae.autoencoder import AutoEncoder

from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler
from megatron.core.models.hunyuan.model import HunyuanParams

from megatron.training import get_args
import torch.nn.functional as F
import torch.distributed as dist
# from my_utils import ForwardProfilerHook
# from my_utils import ModuleProfiler
logger = logging.getLogger(__name__)

def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

class HunyuanPipeline(nn.Module):
    def __init__(self, hunyuan_config,config):
        super().__init__()
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None
        # self.device = params.device
        # params.clip_params['device'] = self.device
        # params.t5_params['device'] = self.device
        #从外面传模型位置进来
        print("Initializing HunyuanPipeline...")

        # pretrained = get_model_path(hunyuan_config.pretrained)
        # vae_pretrained = hunyuan_config.get(
        #     "vae_pretrained", os.path.join(pretrained, "vae")
        # )
        # if mpu.get_pipeline_model_parallel_rank() == 0:
        self.vae = AutoencoderKLHunyuanVideo()
        # profiler_hook = ForwardProfilerHook(start_iter=1, stop_iter=2, rank_only=[0, 2],nvtx_range="VAE forward+DIT foward+backward")
        # profiler_hook.attach(self.vae)


        self.vae.requires_grad_(False)
        #device和dtype怎么传进来
        # self.vae.to(device='cuda', dtype=torch.bfloat16)
        # vae slicing and tiling
        vae_use_slicing = hunyuan_config.vae.get("vae_slicing", False)
        vae_use_tiling = hunyuan_config.vae.get("vae_tiling", False)
        if vae_use_slicing:
            self.vae.enable_slicing()
        if vae_use_tiling:
            self.vae.enable_tiling()
        # print(f"rank {dist.get_rank()} vae dtype is {self.vae.dtype} device = {self.vae.device}")
        hunyuanConfig = HunyuanParams()
        # self.clip_encoder = FrozenCLIPEmbedder(**params.clip_params)
        # self.t5_encoder = FrozenT5Embedder(**params.t5_params)
        latent_channels = self.vae.config.latent_channels

        print("Load HunyuanVideoTransformer3DModel to cuda")
        self.transformer = HunyuanVideoTransformer3DModel(hunyuanConfig,config)
        # self.module_profiler = ModuleProfiler(self.transformer)
        # from my_utils import ForwardProfilerHook
        # profiler_hook = ForwardProfilerHook(start_iter=1, stop_iter=2, rank_only=[0, 2],nvtx_range="DIT forward")
        # profiler_hook.attach(self.transformer)
        print("Loaded HunyuanVideoTransformer3DModel to cuda")


        # self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))
        
        # for name, param in self.transformer.state_dict().items():
        #     if isinstance(param, torch.Tensor):
        #         print(f"Layer: {name}, Shape: {param.shape}")

        
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler()
        self.flow_scheduler_config = hunyuan_config.get("scheduler", dict())
        # self.params = params
        # LoRA setting
        if hunyuan_config.get("lora", False):
            print(hunyuan_config.get("lora", False))
            from peft import (
                LoraConfig,
                get_peft_model_state_dict,
                set_peft_model_state_dict,
            )

            self.transformer.requires_grad_(False)
            lora = hunyuan_config.lora
            transformer_lora_config = LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                init_lora_weights=True,
                target_modules=lora.modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
            self.lora = lora
        
        # 为了模型启动的独立，不要引入get_args,可在config中添加
        # args = get_args()
        # if args.fp16:
        #     self.dtype = torch.float16
        # if args.bf16:
        #     self.dtype = torch.bfloat16
        self.dtype = torch.bfloat16
        print(f"hunyuanpipeline dtype: {self.dtype}")


        self.counter = 0
        
    def forward(self, batch_dict):
        from my_utils import  global_timer
        # global_timer.start("model forward")
        if self.pre_process:
            with torch.no_grad():
                drop_prob = 0
                # latents
                images = batch_dict["images"]
                batch_size, num_frames, _, height, width = images.shape
                # if mpu.get_tensor_context_parallel_rank() == 0:
                #     torch.cuda.cudart().cudaProfilerStart()
                #     torch.cuda.nvtx.range_push('VAE forward')
                
                global_timer.start("VAE forward")
    
                # if dist.get_rank() in [0] :
                    # torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("VAE forward")
                # print(f'Rank {dist.get_rank()} images batch_size = {images.shape[0]}')
                mpu.push_scope("VAE")
                latents = self.forward_vae(images) * self.vae.config.scaling_factor
                
                # if dist.get_rank() in [0] :
                    # torch.cuda.nvtx.range_pop()
                    # torch.cuda.cudart().cudaProfilerStop()
                    
                global_timer.stop("VAE forward")
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.cudart().cudaProfilerStop()
                timesteps = torch.tensor(0, device=torch.cuda.current_device()).long()
                
                # add noise from flow matching scheduler
                scheduler_sigmas = self.flow_scheduler.sigmas.clone()
                weights = compute_density_for_timestep_sampling(
                    weighting_scheme=self.flow_scheduler_config.flow_weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=self.flow_scheduler_config.flow_logit_mean,
                    logit_std=self.flow_scheduler_config.flow_logit_std,
                    mode_scale=self.flow_scheduler_config.flow_mode_scale,
                )
                indices = (weights * self.flow_scheduler.config.num_train_timesteps).long()
                sigmas = scheduler_sigmas[indices].to(device=torch.cuda.current_device())

                timesteps = (sigmas * 1000.0).long()
                broadcast_timesteps(timesteps)

                # torch.manual_seed(22)
                noise = torch.randn(
                    latents.shape,
                    device=torch.cuda.current_device(),
                )

                def expand_tensor_to_dims(tensor, ndim):
                    while len(tensor.shape) < ndim:
                        tensor = tensor.unsqueeze(-1)
                    return tensor

                sigmas = expand_tensor_to_dims(sigmas, ndim=latents.ndim)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

                # embeddings
                prompt_embeds = batch_dict["prompt_embeds"].to(dtype=self.dtype)
                pooled_prompt_embeds = batch_dict["clip_text_embed"].to(dtype=self.dtype)
                prompt_masks = (
                    batch_dict["prompt_masks"].to(dtype=self.dtype)
                    if "prompt_masks" in batch_dict
                    else None
                )

                # conditional_latents
                conditional_latents = None
                if "ref_images" in batch_dict:
                    ref_images = batch_dict["ref_images"]
                    conditional_latents = (
                        self.forward_vae(ref_images) * self.vae.config.scaling_factor
                    )
                    if drop_prob > 0:
                        random_p = torch.rand(batch_size, device=self.device)
                        image_mask = torch.logical_and(
                            random_p >= drop_prob, random_p < 3 * drop_prob
                        )
                        image_mask = 1 - image_mask.float()
                        image_mask = image_mask.to(conditional_latents.dtype)
                        conditional_latents = (
                            conditional_latents * image_mask[:, None, None, None, None]
                        )
                    noisy_model_input = torch.cat(
                        [noisy_model_input, conditional_latents], dim=1
                    )
                if "first_ref_image" in batch_dict:
                    first_ref_image = batch_dict["first_ref_image"]
                    # global_timer.start("first_ref_image VAE")
                    conditional_latents = (
                        self.forward_vae(first_ref_image) * self.vae.config.scaling_factor
                    )
                    # global_timer.start("first_ref_image VAE")
                    pad_size = noisy_model_input.size(2) - conditional_latents.size(2)
                    conditional_latents = F.pad(
                        conditional_latents,
                        (0, 0, 0, 0, 0, pad_size),
                        mode="constant",
                        value=0,
                    )
                    b, c, f, h, w = noisy_model_input.shape
                    mask = torch.zeros((b, 1, f, h, w), device=torch.cuda.current_device())
                    mask[:, :, 0] = 1
                    logger.info(f'Rank {dist.get_rank()}  input = {noisy_model_input.shape}, conditional = {conditional_latents.shape}, mask = {mask.shape}')
                    noisy_model_input = torch.cat(
                        [noisy_model_input, conditional_latents, mask], dim=1
                    )
                if "cn_images" in batch_dict:
                    cn_images = batch_dict["cn_images"].to(self.dtype)
                    cn_images = self.forward_vae(cn_images)
                # TODO guidance check
                guidance_scale = 1.0
                guidance = (
                    torch.tensor(
                        [guidance_scale] * latents.shape[0],
                        dtype=self.dtype,
                        device='cuda',
                    )
                    * 1000.0
                )
                
                # loss
                    # cn model
                if "cn_images" in batch_dict:
                    if hasattr(self.model, "guider"):
                        cn_model = functools.partial(self.model, "guider")
                        cn_images = rearrange(cn_images, "b t c h w -> b c t h w")
                        cn_latents = cn_model(cn_images)
                        # cn_latents = rearrange(cn_latents, "b c t h w -> b t c h w")
                    else:
                        cn_latents = cn_images
                    noisy_model_input = torch.cat([noisy_model_input, cn_latents], dim=1)
                latent_model_input = noisy_model_input.to(torch.bfloat16)
        # dit_timer = MyTimer(use_cuda=True,tag=f"DIT forward dual {os.environ.get('NUM_LAYERS')} single {os.environ.get('NUM_SINGLE_LAYERS')}")
        global_timer.start(f"DIT forward dual {os.environ.get('NUM_LAYERS')}")
        # self.module_profiler.start()
        # if dist.get_rank() in [0] :
        # #     torch.cuda.cudart().cudaProfilerStart()
        #     torch.cuda.nvtx.range_push("DIT forward")
        if os.environ.get("print_shape_info") == "1":
            print("hidden_states.shape:", getattr(latent_model_input, "shape", None))
            print("timestep.shape:", getattr(timesteps, "shape", None))
            print("encoder_hidden_states.shape:", getattr(prompt_embeds, "shape", None))
            print("encoder_attention_mask.shape:", getattr(prompt_masks, "shape", None))
            print("pooled_projections.shape:", getattr(pooled_prompt_embeds, "shape", None))
            print("guidance.shape:", getattr(guidance, "shape", None))
        # print(f"latent device is {latent_model_input.device}")
        # data_for_dit = {
        #     "hidden_states": latent_model_input,
        #     "timesteps": timesteps,
        #     "encoder_hidden_states": prompt_embeds,
        #     # "encoder_attention_mask": prompt_masks,
        #     "pooled_projections": pooled_prompt_embeds,
        #     "guidance": guidance,
        #     "noise": noise,
        #     "latents": latents,
        # }

        # switch scope
        # mpu.pop_scope()
        # mpu.push_scope(stage="DIT", tensors_dict_to_modify = data_for_dit)
        if os.environ["ENABLE_PROFILING_DIT"] == "1":
            from torch.profiler import ProfilerActivity
            def trace_handler(p):
                p.export_chrome_trace(f"torch_prof_{dist.get_rank()}.json")
                with open(f'./prof_summary_rank{dist.get_rank()}.txt', 'w') as f: 
                    f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
            with torch.profiler.profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA, # 如果使用GPU，必须包含 CUDA
                ],
                # schedule=my_schedule,
                on_trace_ready=trace_handler,
                record_shapes=True, # 记录算子的输入张量形状
                profile_memory=True, # 开启内存分析
                with_stack=True # 记录算子的调用栈，方便追溯代码
            ) as prof:
                if self.counter == 1 :
                    print(f"Rank {dist.get_rank()}  nsys CudaProfilerStart()")
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("DIT")
                # model_pred = self.transformer(
                #         hidden_states=data_for_dit["hidden_states"],  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                #         timestep=data_for_dit["timesteps"],  # [263]
                #         encoder_hidden_states=data_for_dit["encoder_hidden_states"],  # [1, 226, 4096]
                #         encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
                #         pooled_projections=data_for_dit["pooled_projections"],  # [1, 1, 768]
                #         guidance=data_for_dit["guidance"],  # []
                #         return_dict=False,
                #     )[0]
                model_pred = self.transformer(
                        hidden_states= latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                        timestep= timesteps,  # [263]
                        encoder_hidden_states= prompt_embeds,  # [1, 226, 4096]
                        encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
                        pooled_projections= pooled_prompt_embeds,  # [1, 1, 768]
                        guidance=guidance,  # []
                        return_dict=False,  
                )[0]
                # 保存 key_averages 表格到文件

                if self.counter == 1 :
                    print(f"Rank {dist.get_rank()}  nsys CudaProfilerStop()")
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
        else: 
            model_pred = self.transformer(
                        hidden_states= latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                        timestep= timesteps,  # [263]
                        encoder_hidden_states= prompt_embeds,  # [1, 226, 4096]
                        encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
                        pooled_projections= pooled_prompt_embeds,  # [1, 1, 768]
                        guidance=guidance,  # []
                        return_dict=False,  
                )[0]        


        # self.module_profiler.stop()
        # self.module_profiler.summary(output_path = "module_profiler.csv")
        global_timer.stop(f"DIT forward dual {os.environ.get('NUM_LAYERS')}")
        
        # if mpu.get_tensor_context_parallel_rank() == 0:
        #     torch.cuda.nvtx.range_pop()
        #     torch.cuda.cudart().cudaProfilerStop()
        # if dist.get_rank() in [0,2]:
        #     torch.cuda.nvtx.range_pop()
        #     torch.cuda.cudart().cudaProfilerStop()
        
        if self.post_process:
            weights = compute_loss_weighting_for_sd3(
                weighting_scheme=self.flow_scheduler_config.flow_weighting_scheme,
                sigmas=sigmas,
            )
            # target = data_for_dit["noise"] - data_for_dit["latents"]
            target = noise - latents

            loss = weights.float() * (model_pred.float() - target.float()).pow(2)
        # global_timer.stop(f"model forward")
        self.counter = self.counter + 1
        return [loss]
    
    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images).latent_dist.sample()
            # torch.manual_seed(42)
            # latents = torch.randn((1, 16, 3, 16, 30), device=torch.cuda.current_device())
        return latents

    def _prepare_rotary_positional_embeddings(self, height, width, num_frames):
        if not self.use_rotary_positional_embeddings:
            return None
        grid_height = height // (self.vae_scale_factor_spatial * self.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.patch_size)
        grid_crops_coords = ((0, 0), (grid_height, grid_width))
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
        freqs_cos = freqs_cos.to(device='cuda', dtype=self.dtype)
        freqs_sin = freqs_sin.to(device='cuda', dtype=self.dtype)
        return freqs_cos, freqs_sin

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)

    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     """Customized load."""
    #     if not isinstance(state_dict, Mapping):
    #         raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    #     missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, False)

    #     if missing_keys is not None:
    #         logger.info(f"Missing keys in state_dict: {missing_keys}.")
    #     if unexpected_keys is not None:
    #         logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
    
    def set_input_tensor(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.transformer.set_input_tensor(input_tensor)
        pass