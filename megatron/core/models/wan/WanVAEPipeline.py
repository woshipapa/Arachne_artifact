
from diffusers import AutoencoderKLWan
import torch
import functools
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


class WanVAEPipeline(nn.Module):

    def __init__(self, hunyuan_config: object):
        """

        Args:
        """
        super().__init__()

        print("Initializing WanVAEPipeline...")
        # pretrained = get_model_path(hunyuan_config.pretrained)
        # vae_pretrained = hunyuan_config.get(
        #     "vae_pretrained", os.path.join(pretrained, "vae")
        # )
        self.vae = AutoencoderKLWan()

        self.vae.requires_grad_(False)

        # self.device = 'cuda'
        # self.dtype = torch.bfloat16
        # self.vae.to(device=self.device, dtype=self.dtype)
        # print(f"[VAEPipeline] Model moved to {self.device} with dtype {self.dtype}.")

        vae_use_slicing = hunyuan_config.vae.get("vae_slicing", False)
        vae_use_tiling = hunyuan_config.vae.get("vae_tiling", False)

        if vae_use_slicing:
            self.vae.enable_slicing()
            print("[WanVAEPipeline] VAE slicing enabled.")
        if vae_use_tiling:
            self.vae.enable_tiling()
            print("[WanVAEPipeline] VAE tiling enabled.")
        # latent_channels = self.vae.config.latent_channels
        # self.scaling_factor = self.vae.config.scaling_factor
        self.dtype = torch.bfloat16
        print("[VAEPipeline] Initialized successfully.")

    def forward_vae(self, images: torch.Tensor) -> torch.Tensor:
        """

        Args:

        Returns:
        """
        images = images.to(self.vae.dtype)
        
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images).latent_dist.sample()

        return latents

    def forward(self, batch_dict: dict) -> dict:
        main_latents = None
        encoded_outputs = {}
        self.device = torch.cuda.current_device()
        if "images" in batch_dict:
            from my_utils import global_timer
            import os
            bs = os.environ.get('bs')
            frames = os.environ.get('frames')
            height = os.environ.get('height')
            width = os.environ.get('width')
            sp = os.environ.get('sp')
            log_str = f"vae_forward_bs_{bs}_f_{frames}_h_{height}_w_{width}_sp_{sp}_dist{os.environ.get('PROFILE_TAG')}"
            global_timer.start(log_str)
            main_latents = self.forward_vae(batch_dict["images"]) 
            encoded_outputs["latents"] = main_latents
            global_timer.stop(log_str)

            # conditional_latents
        # conditional_latents_list = []
        if "ref_images" in batch_dict:
            ref_images = batch_dict["ref_images"]
            ref_latents = self.forward_vae(ref_images) 
            encoded_outputs["ref_latents"] = ref_latents
            # conditional_latents_list.append(ref_latents)

        if "first_ref_image" in batch_dict:
            if main_latents is None:
                raise ValueError(
                    "Cannot process 'first_ref_image' without 'images' to determine target shape."
                )

            first_ref_image = batch_dict["first_ref_image"].to(
                device=torch.cuda.current_device()
            )
            first_ref_latents = self.forward_vae(first_ref_image) 

            pad_size = main_latents.size(2) - first_ref_latents.size(2)
            padded_ref_latents = F.pad(
                first_ref_latents, (0, 0, 0, 0, 0, pad_size), mode="constant", value=0
            )

            b, c, f, h, w = main_latents.shape
            mask = torch.zeros((b, 1, f, h, w), device=self.device, dtype=self.dtype)
            mask[:, :, 0] = 1

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

        return encoded_outputs
