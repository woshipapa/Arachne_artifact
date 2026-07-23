import copy
import functools
import os
from einops import rearrange
import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
import json
from torch import nn
import torch.nn.functional as F


from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler

from vast.models import GuiderModel, HunyuanVideoTransformer3DModel

from vast.models import utils as gm_utils
from vast.models import ModuleDict

from vast.train import Trainer
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class HunYuanVideoTrainer(Trainer):
    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        model = dict()
        # vae
        vae_pretrained = model_config.get(
            "vae_pretrained", os.path.join(pretrained, "vae")
        )
        vae_pretrained = gm_utils.get_model_path(vae_pretrained)
        with self.timeit_context("Load VAE"):
            self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_pretrained,
                trust_remote_code=True,
                # disable_mmap=True,
            )
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
            transformer_pretrained = model_config.get(
                "transformer_pretrained", os.path.join(pretrained, "transformer")
            )
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                transformer_pretrained,
                allow_pickle=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                # disable_mmap=True,
            )
        with self.timeit_context("Process Model"):
            latent_channels = self.vae.config.latent_channels
            transformer = process_model(
                transformer, latent_channels, model_config.get("transformer", dict())
            )

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
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler()
        self.flow_scheduler_config = model_config.get("scheduler", dict())

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
            drop_prob = 0
            transformer = functools.partial(self.model, "transformer")
            # latents
            images = batch_dict["images"]
            batch_size, num_frames, _, height, width = images.shape
            latents = self.forward_vae(images) * self.vae.config.scaling_factor
            # add noise from flow matching scheduler
            scheduler_sigmas = self.flow_scheduler.sigmas.clone().to(
                device=self.accelerator.device
            )
            weights = compute_density_for_timestep_sampling(
                weighting_scheme=self.flow_scheduler_config.flow_weighting_scheme,
                batch_size=batch_size,
                logit_mean=self.flow_scheduler_config.flow_logit_mean,
                logit_std=self.flow_scheduler_config.flow_logit_std,
                mode_scale=self.flow_scheduler_config.flow_mode_scale,
            )
            indices = (weights * self.flow_scheduler.config.num_train_timesteps).long()
            sigmas = scheduler_sigmas[indices]
            timesteps = (sigmas * 1000.0).long()
            noise = torch.randn(
                latents.shape,
                device=self.accelerator.device,
            )

            def expand_tensor_to_dims(tensor, ndim):
                while len(tensor.shape) < ndim:
                    tensor = tensor.unsqueeze(-1)
                return tensor

            sigmas = expand_tensor_to_dims(sigmas, ndim=latents.ndim)
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
            # embeddings
            prompt_embeds = batch_dict["prompt_embeds"].to(self.dtype)
            pooled_prompt_embeds = batch_dict["clip_text_embed"].to(self.dtype)
            prompt_masks = (
                batch_dict["prompt_masks"].to(self.dtype)
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
                conditional_latents = (
                    self.forward_vae(first_ref_image) * self.vae.config.scaling_factor
                )
                pad_size = noisy_model_input.size(2) - conditional_latents.size(2)
                conditional_latents = F.pad(
                    conditional_latents,
                    (0, 0, 0, 0, 0, pad_size),
                    mode="constant",
                    value=0,
                )
                b, c, f, h, w = noisy_model_input.shape
                mask = torch.zeros((b, 1, f, h, w), device=self.accelerator.device)
                mask[:, :, 0] = 1
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
                    device=self.device,
                )
                * 1000.0
            )

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

            model_pred = transformer(
                hidden_states=latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                timestep=timesteps,  # [263]
                encoder_hidden_states=prompt_embeds,  # [1, 226, 4096]
                encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
                pooled_projections=pooled_prompt_embeds,  # [1, 1, 768]
                guidance=guidance,  # []
                return_dict=False,
            )[0]
        weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.flow_scheduler_config.flow_weighting_scheme,
            sigmas=sigmas,
        )
        target = noise - latents
        loss = weights.float() * (model_pred.float() - target.float()).pow(2)
        loss = loss.mean()
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
