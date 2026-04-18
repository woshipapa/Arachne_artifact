
from megatron.core import mpu

from .model import CogVideoXParams, CogVideoXTransformer3DModel
from .cogvideox_loss import CogVideoXLoss
import torch
import torch.nn as nn
import os
from diffusers.models.embeddings import get_3d_rotary_pos_embed
class CogDiTPipeline(nn.Module):
    """
    专门负责DiT（扩散型Transformer）的管线。
    这是模型训练的核心部分，在潜空间中进行去噪。
    """

    def __init__(self, cogvideox_config: object, config: object):
        """
        初始化DiT管线。
        严格遵循原始HunyuanPipeline的初始化逻辑。

        Args:
            hunyuan_config (object): global_config.models中的配置。
            config (object): 传递给Transformer模型的特定配置。
        """
        super().__init__()
        print("[CogDiTPipeline] Initializing...")

        # 1. 设置流水线并行相关的标志位
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()

        # 2. 初始化文本编码器 (这部分在您的原代码中被注释，这里作为可选模块保留)
        # if hasattr(hunyuan_config, "clip_params"):
        #     self.clip_encoder = FrozenCLIPEmbedder(**hunyuan_config.clip_params)
        # if hasattr(hunyuan_config, "t5_params"):
        #     self.t5_encoder = FrozenT5Encoder(**hunyuan_config.t5_params)

        # 3. 初始化核心的Transformer模型 (DiT)
        # HunyuanParams的创建和Transformer的初始化都来自原始代码
        cogConfig = CogVideoXParams()
        self.transformer = CogVideoXTransformer3DModel(cogConfig, config)
        print("[DiTPipeline] CogVideoXTransformer3DModel created.")

        # 4. 初始化扩散过程的调度器 (Scheduler)
        self.flow_scheduler = CogVideoXLoss(
            world_size=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            **cogvideox_config.get("loss", dict()),
        )
        self.flow_scheduler_config = cogvideox_config.get("scheduler", dict())

        # 5. 配置LoRA (如果需要)
        # 这部分逻辑完全复制自原始代码
        if cogvideox_config.get("lora", False):
            from peft import LoraConfig

            print("[DiTPipeline] Configuring LoRA...")
            self.transformer.requires_grad_(False)
            lora = cogvideox_config.lora
            transformer_lora_config = LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                init_lora_weights=True,
                target_modules=lora.modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
            self.lora = lora
            print("[DiTPipeline] LoRA configured successfully.")
        self.dtype = torch.bfloat16
        self.patch_size = cogConfig.patch_size
        self.use_rotary_positional_embeddings = cogConfig.use_rotary_positional_embeddings

        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        # 6. 设置数据类型
        # self.dtype = torch.bfloat16 # 遵循原始代码的硬编码
        # self.device = 'cuda' # 假设默认设备
        # self.to(self.device)
        # print(f"[DiTPipeline] Model moved to {self.device} with dtype {self.dtype}.")

        self.counter = 0


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

    def forward(self, batch_dict: dict, encoded_bundle: dict) -> torch.Tensor:
        """
        执行一个完整的训练步骤：加噪、模型预测、计算损失。
        此方法封装了原始 `HunyuanPipeline.forward` 中所有与DiT相关的逻辑。

        Args:
            batch_dict (dict): 包含文本嵌入、掩码等信息的字典。
            latents (torch.Tensor): 已经由VAEPipeline编码好的干净的潜空间向量。

        Returns:
            torch.Tensor: 计算出的损失值张量。
        """
        # 流水线第一阶段：准备Transformer的输入
        if self.pre_process:
            with torch.no_grad():

                # bfchw
                latents = encoded_bundle["latents"]
                height = latents.shape[3] * 8
                width = latents.shape[4] * 8
                os.environ["bs"] = str(latents.shape[0])
                os.environ["height"] = str(height)
                os.environ["width"] = str(width)    
                os.environ["frames"] = str((latents.shape[1] - 1) * 4 + 1)
                os.environ['sp'] = str(mpu.get_context_parallel_world_size())
                batch_size = latents.shape[0]
                # timesteps = torch.tensor(0, device=torch.cuda.current_device()).long()
                # a. 加噪过程
                noisy_model_input, timesteps = self.flow_scheduler.add_noise(latents)

                # 准备文本嵌入
                prompt_embeds = batch_dict["prompt_embeds"].to(dtype=self.dtype)
                pooled_prompt_embeds = batch_dict["clip_text_embed"].to(
                    dtype=self.dtype
                )
                prompt_masks = (
                    batch_dict["prompt_masks"].to(dtype=self.dtype)
                    if "prompt_masks" in batch_dict
                    else None
                )
                image_rotary_emb = self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1)
            )
        from my_utils import global_timer

        # global_timer.start(f"DIT forward dual {os.environ.get('NUM_LAYERS')}")
        # global_timer.start(
        #     f"DIT forward dual {os.environ.get('NUM_LAYERS')}, {os.environ.get('SP_SETTING')}"
        # )
        forward_str = f"forward_single{os.environ.get('NUM_LAYERS')}_bs_{os.environ.get('bs')}_f_{os.environ.get('frames')}_h_{os.environ.get('height')}_w_{os.environ.get('width')}_sp{os.environ.get('sp')}"
        # print(f"[Rank {dist.get_rank()}]======enter transformer==============") # no
        global_timer.start(forward_str)
        model_pred = self.transformer(
            hidden_states=noisy_model_input.to(self.dtype),  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
            timestep=timesteps.to(self.dtype),  # [263]
            encoder_hidden_states=prompt_embeds,  # [1, 226, 4096]
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        global_timer.stop(forward_str)


        loss = self.flow_scheduler(model_pred)

        self.counter += 1
        # 在流水线并行中，只有最后一个stage会返回一个有效的loss张量
        return loss

    # --- 其他辅助方法，完全复制自原始代码 ---

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """为保存检查点提供transformer的状态字典。"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        """加载状态字典到transformer。"""
        # 注意：原代码中此方法被注释，这里我们提供一个实现
        return self.transformer.load_state_dict(state_dict, strict)

    def set_input_tensor(self, input_tensor):
        """用于流水线并行的输入设置。"""
        # self.transformer.set_input_tensor(input_tensor)
        pass
