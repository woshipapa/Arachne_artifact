import copy
import functools
import os
from einops import rearrange
import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
import json
from torch import nn
import torch.nn.functional as F

# from vast.models import autoencoder_kl_hunyuan_video
from diffusers import AutoencoderKLWan
# from diffusers import FlowMatchEulerDiscreteScheduler

from vast.models import GuiderModel, WanModel, WanParams
from .flow_match import FlowMatchScheduler
from vast.models import utils as gm_utils
from vast.models import ModuleDict

from vast.train import Trainer
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class WanTrainer(Trainer):
    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        model = dict()
        # vae
        vae_pretrained = model_config.get(
            "vae_pretrained", os.path.join(pretrained, "vae")
        )
        vae_pretrained = gm_utils.get_model_path(vae_pretrained)
        with self.timeit_context("Load VAE"):
            # self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            #     vae_pretrained,
            #     trust_remote_code=True,
            #     # disable_mmap=True,
            # )
            self.vae = AutoencoderKLWan()
            self.vae.requires_grad_(False)
            self.vae.to(self.device, dtype=self.dtype)
            # vae slicing and tiling
            vae_use_slicing = model_config.vae.get("vae_slicing", False)
            vae_use_tiling = model_config.vae.get("vae_tiling", False)
            if vae_use_slicing:
                self.vae.enable_slicing()
            if vae_use_tiling:
                self.vae.enable_tiling()

        # transformer
        with self.timeit_context("Load Transformer"):
            # transformer_pretrained = model_config.get(
            #     "transformer_pretrained", os.path.join(pretrained, "transformer")
            # )
            # transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            #     transformer_pretrained,
            #     allow_pickle=True,
            #     trust_remote_code=True,
            #     torch_dtype=torch.bfloat16,
            #     # disable_mmap=True,
            # )
            transformer = WanModel(WanParams())

        # input_channels = 16 no 33    
        # with self.timeit_context("Process Model"):
        #     latent_channels = self.vae.config.latent_channels
        #     transformer = process_model(
        #         transformer, latent_channels, model_config.get("transformer", dict())
        #     )

        # LoRA setting
        self.lora = False
        if model_config.get("lora", False):
            from peft import (
                LoraConfig,
                get_peft_model_state_dict,
                set_peft_model_state_dict,
            )

            transformer.requires_grad_(False)
            lora = model_config.lora
            transformer_lora_config = LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                init_lora_weights=True,
                target_modules=lora.modules,
            )
            transformer.add_adapter(transformer_lora_config)
            self.lora = lora

        with self.timeit_context(f"transformer to {self.device}"):
            transformer.to(self.device)
        with self.timeit_context("model.update transformer"):
            model.update(transformer=transformer)

        # flow scheduler
        self.flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.flow_scheduler.set_timesteps(1000, training=True)
        self.flow_scheduler_config = model_config.get("scheduler", dict())
        # self.dtype = torch.bfloat16
        with self.accelerator.autocast():
            # cn model
            if hasattr(model_config, "guider"):
                with self.timeit_context("Load Guider Model"):
                    cn_model = GuiderModel(model_config.guider)
                    cn_model.to(self.device)
                    model.update(guider=cn_model)
            # model
            with self.timeit_context("Load Checkpoint"):
                checkpoint = model_config.get("checkpoint", None)
                strict = model_config.get("strict", True)
                self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
            with self.timeit_context(f"model.to({self.device})"):
                model = ModuleDict(model)
                model.to(self.device)
                model.train()
        return model

    def forward_step(self, batch_dict):
        with self.accelerator.autocast():
            # from utils import logger
            drop_prob = 0
            transformer = functools.partial(self.model, "transformer")
            # latents
            images = batch_dict["images"]
            batch_size, num_frames, _, height, width = images.shape
            # WanVAEPipeline
            latents = self.forward_vae(images) 
            
            encoded_outputs = {}
            if "ref_images" in batch_dict:
                ref_images = batch_dict["ref_images"]
                ref_latents = self.forward_vae(ref_images) 
                encoded_outputs["ref_latents"] = ref_latents
                # conditional_latents_list.append(ref_latents)

            # b. 处理 first_ref_image
            if "first_ref_image" in batch_dict:
                if latents is None:
                    raise ValueError(
                        "Cannot process 'first_ref_image' without 'images' to determine target shape."
                    )

                first_ref_image = batch_dict["first_ref_image"].to(
                    device=torch.cuda.current_device()
                )
                first_ref_latents = self.forward_vae(first_ref_image) 

                # 执行与原始代码一致的填充逻辑
                pad_size = latents.size(2) - first_ref_latents.size(2)
                padded_ref_latents = F.pad(
                    first_ref_latents, (0, 0, 0, 0, 0, pad_size), mode="constant", value=0
                )

                # 创建对应的mask
                b, c, f, h, w = latents.shape
                mask = torch.zeros((b, 1, f, h, w), device=self.device, dtype=self.dtype)
                mask[:, :, 0] = 1  # 在第一帧位置标记为1

                # 将填充后的latents和mask都存入输出字典，由DiT管线决定如何拼接
                encoded_outputs["first_ref_latents"] = padded_ref_latents
                encoded_outputs["first_ref_mask"] = mask
                # conditional_latents_list.append(padded_ref_latents)
                # conditional_latents_list.append(mask)
            if "cn_images" in batch_dict:
                cn_images = batch_dict["cn_images"].to(self.dtype)
                cn_images = self.forward_vae(cn_images)
                if hasattr(self.model, "guider"):
                    cn_model = functools.partial(self.model, "guider")
                    cn_images = rearrange(cn_images, "b t c h w -> b c t h w")
                    cn_latents = cn_model(cn_images)
                    # cn_latents = rearrange(cn_latents, "b c t h w -> b t c h w")
                else:
                    cn_latents = cn_images
                # conditional_latents_list.append(cn_latents)
                encoded_outputs["cn_latents"] = cn_latents




            # add noise from flow matching scheduler
            # --- 2. 加噪 (来自 WanDiTPipeline.forward) ---
            # 注意: 这里我们使用更简单的 timestep 采样，与您的 DITPipeline 一致
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.flow_scheduler.num_train_timesteps, (1,))
            timesteps = self.flow_scheduler.timesteps[timestep_id].to(
                dtype = self.dtype, device = torch.cuda.current_device()
            )

            # 获取加噪后的 latents 和训练目标
            noisy_model_input = self.flow_scheduler.add_noise(latents, noise, timesteps)
            training_target = self.flow_scheduler.training_target(latents, noise, timesteps)
            


            # logger.info(f'noisy_model_input shape is {noisy_model_input.shape}')
            # embeddings
            prompt_embeds = batch_dict["prompt_embeds"].to(self.dtype)
            # print(f'prompt_embeds is {prompt_embeds}, shape is {prompt_embeds.shape}')
            pooled_prompt_embeds = batch_dict["clip_text_embed"].to(self.dtype)
            prompt_masks = (
                batch_dict["prompt_masks"].to(self.dtype)
                if "prompt_masks" in batch_dict
                else None
            )
            # conditional_latents
          
        # noisy_model_input = F.pad(noisy_model_input,(0,0,0,0,0,1),mode='replicate')
        # print(f'latents shape is {noisy_model_input.shape}') 33 channel
        # loss
        with self.accelerator.autocast():
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
            # logger.info(f'input transformer block {latent_model_input.shape}, prompt_embeds {prompt_embeds.shape}')
            model_pred = transformer(
                x=latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                timestep=timesteps,  # [263]
                context=prompt_embeds,  # [1, 226, 4096]
                # encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
                # pooled_projections=pooled_prompt_embeds,  # [1, 1, 768]
                # guidance=guidance,  # []
                # return_dict=False,
                clip_feature=None,
                y=None,
            )
            loss = F.mse_loss(model_pred.float(), training_target.float())
        
            
            
            # 确保权重形状可以广播
            loss = loss * self.flow_scheduler.training_weight(timesteps)
            
            
            return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images).latent_dist.sample()
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
        freqs_cos = freqs_cos.to(device=self.device, dtype=self.dtype)
        freqs_sin = freqs_sin.to(device=self.device, dtype=self.dtype)
        return freqs_cos, freqs_sin


def process_model(model, latent_channels, cfg):
    in_channels = cfg.get("in_channels", latent_channels)
    out_channels = cfg.get("out_channels", latent_channels)

    if model.config.in_channels != in_channels:
        model_config = copy.deepcopy(model.config)
        model_config["in_channels"] = in_channels
        model_config["out_channels"] = out_channels
        new_model = HunyuanVideoTransformer3DModel.from_config(model_config)
        state_dict = model.state_dict()
        if model.config.in_channels != in_channels:
            weight = state_dict["x_embedder.proj.weight"]
            if model.config.in_channels > in_channels:
                new_weight = weight[:, :in_channels]
            else:
                weight_shape = list(weight.shape)
                weight_shape[1] = in_channels
                new_weight = weight.new_zeros(weight_shape)
                new_weight[:, : weight.shape[1]] = weight
            state_dict["x_embedder.proj.weight"] = new_weight
        new_model.load_state_dict(state_dict, strict=False)
        del model
        model = new_model
    return model
