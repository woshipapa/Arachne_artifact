import copy
import functools
import os
from einops import rearrange
import torch
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.schedulers import CogVideoXDDIMScheduler
from vast.models import CogVideoXTransformer3DModel, GuiderModel
from vast.models import utils as gm_utils
from vast.models import ModuleDict

from vast.train import Trainer
from .cogvideox_loss import CogVideoXLoss


class CogVideoXTrainer(Trainer):
    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        model = dict()
        # vae
        vae_pretrained = model_config.get(
            "vae_pretrained", os.path.join(pretrained, "vae")
        )
        vae_pretrained = gm_utils.get_model_path(vae_pretrained)
        with self.timeit_context("Load VAE"):
            # self.vae = AutoencoderKLCogVideoX.from_pretrained(vae_pretrained)
            self.vae = AutoencoderKLCogVideoX()
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
            # transformer = CogVideoXTransformer3DModel.from_pretrained(
            #     transformer_pretrained
            # )
            transformer = CogVideoXTransformer3DModel()
        # with self.timeit_context("Process Model"):
        #     latent_channels = self.vae.config.latent_channels
        #     transformer = process_model(
        #         transformer, latent_channels, model_config.get("transformer", dict())
        #     )
        with self.timeit_context(f"transformer to {self.device}"):
            transformer.to(self.device)
        with self.timeit_context("model.update transformer"):
            model.update(transformer=transformer)

        with self.accelerator.autocast():
            # cn model
            if hasattr(model_config, "guider"):
                cn_model = GuiderModel(model_config.guider)
                cn_model.to(self.device)
                model.update(guider=cn_model)
            # loss
            self.loss_func = CogVideoXLoss(
                world_size=self.num_processes, rank=self.process_index, **model_config.loss
            )
            # model
            checkpoint = model_config.get("checkpoint", None)
            strict = model_config.get("strict", True)
            self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
            model = ModuleDict(model)
            model.to(self.device)
            model.train()
            self.vae_scale_factor_spatial = 2 ** (
                len(self.vae.config.block_out_channels) - 1
            )
            self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio
            self.patch_size = transformer.config.patch_size
            self.use_rotary_positional_embeddings = (
                transformer.config.use_rotary_positional_embeddings
            )
            self.attention_head_dim = transformer.config.attention_head_dim
        return model

    def forward_step(self, batch_dict):
        with self.accelerator.autocast():
            from my_utils import global_timer
            drop_prob = 0
            transformer = functools.partial(self.model, "transformer")
            # latents
            images = batch_dict["images"]
            batch_size, num_frames, _, height, width = images.shape
            global_timer.start(f"cogvideox_vae_forward_bs_{batch_size}_f_{num_frames}_h_{height}_w_{width}")
            latents = self.forward_vae(images) * self.vae.config.scaling_factor
            global_timer.stop(f"cogvideox_vae_forward_bs_{batch_size}_f_{num_frames}_h_{height}_w_{width}")
            # add noise
            noisy_model_input, timesteps = self.loss_func.add_noise(latents)
            # embeddings
            if "prompt_embeds" in batch_dict:
                embeddings = batch_dict["prompt_embeds"].to(self.dtype)
            else:
                assert False
            # conditional_latents
            conditional_latents = None
            if "ref_images" in batch_dict:
                ref_images = batch_dict["ref_images"]
                conditional_latents = self.forward_vae(ref_images)
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
                    [noisy_model_input, conditional_latents], dim=2
                )

            if "cn_images" in batch_dict:
                cn_images = batch_dict["cn_images"].to(self.dtype)
                cn_images = self.forward_vae(cn_images)

        # loss
        with self.accelerator.autocast():
            # cn model
            if "cn_images" in batch_dict:
                if hasattr(self.model, "guider"):
                    cn_model = functools.partial(self.model, "guider")
                    cn_images = rearrange(cn_images, "b t c h w -> b c t h w")
                    cn_latents = cn_model(cn_images)
                    cn_latents = rearrange(cn_latents, "b c t h w -> b t c h w")
                else:
                    cn_latents = cn_images
                noisy_model_input = torch.cat([noisy_model_input, cn_latents], dim=2)
            image_rotary_emb = self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1)
            )
            dit_log = f"cogvideox_dit_forward_bs_{noisy_model_input.shape[0]}_f_{noisy_model_input.shape[2]}_h_{noisy_model_input.shape[3]}_w_{noisy_model_input.shape[4]}"
            global_timer.start(dit_log)
            model_pred = transformer(
                hidden_states=noisy_model_input.to(self.dtype),
                encoder_hidden_states=embeddings,
                timestep=timesteps.to(self.dtype),
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            global_timer.stop(dit_log)
        loss = self.loss_func(model_pred)
        return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            num_sample_frames_batch_size = self.vae.num_sample_frames_batch_size
            self.vae.num_sample_frames_batch_size = min(
                num_sample_frames_batch_size, images.shape[2]
            )
            latents = self.vae.encode(images).latent_dist.sample()
            self.vae.num_sample_frames_batch_size = num_sample_frames_batch_size
            latents = rearrange(latents, "b c f h w -> b f c h w")
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
        new_model = CogVideoXTransformer3DModel.from_config(model_config)
        state_dict = model.state_dict()
        if model.config.in_channels != in_channels:
            weight = state_dict["patch_embed.proj.weight"]
            if model.config.in_channels > in_channels:
                new_weight = weight[:, :in_channels]
            else:
                weight_shape = list(weight.shape)
                weight_shape[1] = in_channels
                new_weight = weight.new_zeros(weight_shape)
                new_weight[:, : weight.shape[1]] = weight
            state_dict["patch_embed.proj.weight"] = new_weight
        new_model.load_state_dict(state_dict, strict=False)
        del model
        model = new_model
    return model
