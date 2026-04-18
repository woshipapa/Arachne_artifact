
from diffusers import AutoencoderKLWan
import torch
import functools
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


class WanVAEPipeline(nn.Module):
    """
    专门负责VAE（变分自编码器）的管线。
    此版本严格遵循原始HunyuanPipeline的初始化风格。
    """

    def __init__(self, hunyuan_config: object):
        """
        初始化VAE管线。

        Args:
            hunyuan_config (object): 包含所有配置的顶层对象，与原始pipeline保持一致。
        """
        super().__init__()

        # 1. 初始化VAE模型
        # 完全遵循原始代码的写法
        print("Initializing WanVAEPipeline...")
        # pretrained = get_model_path(hunyuan_config.pretrained)
        # vae_pretrained = hunyuan_config.get(
        #     "vae_pretrained", os.path.join(pretrained, "vae")
        # )
        self.vae = AutoencoderKLWan()

        # 2. VAE权重通常是冻结的
        self.vae.requires_grad_(False)

        # 3. 设置设备和数据类型
        # 原始代码中这部分是待定项，我们在此遵循其硬编码的默认值
        # 在实际应用中，将这些值加入config会更灵活
        # self.device = 'cuda'
        # self.dtype = torch.bfloat16
        # self.vae.to(device=self.device, dtype=self.dtype)
        # print(f"[VAEPipeline] Model moved to {self.device} with dtype {self.dtype}.")

        # 4. 配置VAE的slicing和tiling，严格使用 hunyuan_config.vae 的访问路径
        vae_use_slicing = hunyuan_config.vae.get("vae_slicing", False)
        vae_use_tiling = hunyuan_config.vae.get("vae_tiling", False)

        if vae_use_slicing:
            self.vae.enable_slicing()
            print("[WanVAEPipeline] VAE slicing enabled.")
        if vae_use_tiling:
            self.vae.enable_tiling()
            print("[WanVAEPipeline] VAE tiling enabled.")
        # latent_channels = self.vae.config.latent_channels
        # 5. 保存与DiT交互所需的关键参数
        # self.scaling_factor = self.vae.config.scaling_factor
        self.dtype = torch.bfloat16
        print("[VAEPipeline] Initialized successfully.")

    def forward_vae(self, images: torch.Tensor) -> torch.Tensor:
        """
        将输入的像素值（视频帧）编码为潜空间表示。
        此方法严格复制自原始 `HunyuanPipeline.forward_vae`。

        Args:
            images (torch.Tensor): 形状为 B, F, C, H, W 的视频张量。

        Returns:
            torch.Tensor: 未经缩放的潜空间张量。调用者需要手动乘以 self.scaling_factor。
        """
        images = images.to(self.vae.dtype)
        # 原始代码中的打印语句，予以保留
        
        with torch.no_grad():
            # VAE模型需要 B, C, F, H, W 的格式
            images = rearrange(images, "b f c h w -> b c f h w")
            # 编码并从分布中采样
            latents = self.vae.encode(images).latent_dist.sample()

        # 注意：这里不执行缩放。与原始代码一致，缩放操作由外部逻辑处理。
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
            # 编码后立刻进行缩放，这与原始代码逻辑一致
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

        # b. 处理 first_ref_image
        if "first_ref_image" in batch_dict:
            if main_latents is None:
                raise ValueError(
                    "Cannot process 'first_ref_image' without 'images' to determine target shape."
                )

            first_ref_image = batch_dict["first_ref_image"].to(
                device=torch.cuda.current_device()
            )
            first_ref_latents = self.forward_vae(first_ref_image) 

            # 执行与原始代码一致的填充逻辑
            pad_size = main_latents.size(2) - first_ref_latents.size(2)
            padded_ref_latents = F.pad(
                first_ref_latents, (0, 0, 0, 0, 0, pad_size), mode="constant", value=0
            )

            # 创建对应的mask
            b, c, f, h, w = main_latents.shape
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

        return encoded_outputs
