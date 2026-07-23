import copy
import functools
import os
from einops import rearrange
import torch
from diffusers.models import AutoencoderKLCogVideoX
from einops import rearrange
from vast.models import VASTTransformerModel, GuiderModel
from vast.models import utils as gm_utils
from vast.models import ModuleDict

from vast.train import Trainer
from .vastbase_loss import VASTBASELoss

class VASTBASETrainer(Trainer):
    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        model = dict()
        # vae
        vae_pretrained = model_config.get('vae_pretrained', os.path.join(pretrained, 'vae'))
        vae_pretrained = gm_utils.get_model_path(vae_pretrained)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(vae_pretrained)
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.vae.to(self.device, dtype=self.dtype)
        # transformer
        transformer_pretrained = model_config.get('transformer_pretrained', os.path.join(pretrained, 'transformer'))
        from_giga = model_config.get('from_giga', False)
        patch_size = model_config.get('patch_size', [(1,2,2),(1,4,4)])
        only_orgin = model_config.get('only_orgin', False)
        transformer = VASTTransformerModel.from_pretrained(
            transformer_pretrained, 
            torch_dtype=torch.bfloat16,
            from_giga=from_giga,
            patch_size=patch_size,
            only_orgin=only_orgin,
            )
        transformer.to(self.device)
        transformer._set_gradient_checkpointing(transformer, value=True)
        model.update(transformer=transformer)
        # cn model
        if hasattr(model_config, 'guider'):
            cn_model = GuiderModel(model_config.guider)
            cn_model.to(self.device)
            model.update(guider=cn_model)
        # loss
        self.loss_func = VASTBASELoss(world_size=self.num_processes, rank=self.process_index, **model_config.loss)
        # model
        checkpoint = model_config.get('checkpoint', None)
        strict = model_config.get('strict', True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        for name, param in model.named_parameters():
            if model_config.get("trainable_params", "")=="":
                param.requires_grad = True
            else:
                param.requires_grad = False
                for p in model_config.get("trainable_params", ""):
                    if p in name:
                        param.requires_grad = True
        print("Trainable parameters:*******************\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        # import pdb; pdb.set_trace()
        model.to(self.device)
        model.train()
        return model

    def forward_step(self, batch_dict):
        drop_prob = 0
        transformer = functools.partial(self.model, 'transformer')
        # latents
        images = batch_dict['images']
        batch_size, num_frames, _, height, width = images.shape
        latents = self.forward_vae(images) * self.vae.config.scaling_factor
        # add noise
        noisy_model_input, timesteps = self.loss_func.add_noise(latents)
        # embeddings
        if 'prompt_embeds' in batch_dict:
            embeddings = batch_dict['prompt_embeds'].to(self.dtype)
        else:
            assert False
        # conditional_latents
        conditional_latents = None
        if 'ref_images' in batch_dict:
            ref_images = batch_dict['ref_images']
            conditional_latents = self.forward_vae(ref_images)
            if drop_prob > 0:
                random_p = torch.rand(batch_size, device=self.device)
                image_mask = torch.logical_and(random_p >= drop_prob, random_p < 3 * drop_prob)
                image_mask = 1 - image_mask.float()
                image_mask = image_mask.to(conditional_latents.dtype)
                conditional_latents = conditional_latents * image_mask[:, None, None, None, None]
            noisy_model_input = torch.cat([noisy_model_input, conditional_latents], dim=2)
        
        if 'cn_images' in batch_dict:
            cn_images = batch_dict['cn_images'].to(self.dtype)
            cn_images = self.forward_vae(cn_images)
       
        # loss
        with self.accelerator.autocast():
            # cn model
            if 'cn_images' in batch_dict:
                if hasattr(self.model, 'guider'):
                    cn_model = functools.partial(self.model, 'guider')
                    cn_images = rearrange(cn_images, 'b t c h w -> b c t h w')
                    cn_latents = cn_model(cn_images)
                    cn_latents = rearrange(cn_latents, 'b c t h w -> b t c h w')
                else:
                    cn_latents = cn_images
                noisy_model_input = torch.cat([noisy_model_input, cn_latents], dim=2)
            model_pred = transformer(
                hidden_states=noisy_model_input.to(self.dtype),
                encoder_hidden_states=embeddings,
                timestep=timesteps.to(self.dtype),
                return_dict=False,
            )[0]
        # import pdb; pdb.set_trace()
        loss = self.loss_func(model_pred)
        return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b f c h w -> b c f h w')
            num_sample_frames_batch_size = self.vae.num_sample_frames_batch_size
            self.vae.num_sample_frames_batch_size = min(num_sample_frames_batch_size, images.shape[2])
            latents = self.vae.encode(images).latent_dist.sample()
            self.vae.num_sample_frames_batch_size = num_sample_frames_batch_size
            latents = rearrange(latents, 'b c f h w -> b f c h w')
        return latents


