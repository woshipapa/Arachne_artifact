import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu
import torch.distributed as dist

# 导入所有需要的模块，请确保这些路径在您的项目中是正确的
from megatron.core.models.hunyuan.model import (
    HunyuanVideoTransformer3DModel,
    HunyuanParams,
)
from megatron.core.models.sampler.flow_matching.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
import os
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# 假设文本编码器和LoRA相关模块可导入
# from your_model_library import FrozenCLIPEmbedder, FrozenT5Encoder, LoraConfig


def broadcast_timesteps(input: torch.Tensor):
    """
    一个辅助函数，用于在张量-上下文并行组中广播时间步。
    完全复制自原始代码。
    """
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(
            input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group()
        )


class DiTPipeline(nn.Module):
    """
    专门负责DiT（扩散型Transformer）的管线。
    这是模型训练的核心部分，在潜空间中进行去噪。
    """

    def __init__(self, hunyuan_config: object, config: object):
        """
        初始化DiT管线。
        严格遵循原始HunyuanPipeline的初始化逻辑。

        Args:
            hunyuan_config (object): 包含所有配置的顶层对象。
            config (object): 传递给Transformer模型的特定配置。
        """
        super().__init__()
        print("[DiTPipeline] Initializing...")

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
        hunyuanConfig = HunyuanParams()
        self.transformer = HunyuanVideoTransformer3DModel(hunyuanConfig, config)
        print("[DiTPipeline] HunyuanVideoTransformer3DModel created.")

        # 4. 初始化扩散过程的调度器 (Scheduler)
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler()
        self.flow_scheduler_config = hunyuan_config.get("scheduler", dict())

        # 5. 配置LoRA (如果需要)
        # 这部分逻辑完全复制自原始代码
        if hunyuan_config.get("lora", False):
            from peft import LoraConfig

            print("[DiTPipeline] Configuring LoRA...")
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
            print("[DiTPipeline] LoRA configured successfully.")
        self.dtype = torch.bfloat16
        # 6. 设置数据类型
        # self.dtype = torch.bfloat16 # 遵循原始代码的硬编码
        # self.device = 'cuda' # 假设默认设备
        # self.to(self.device)
        # print(f"[DiTPipeline] Model moved to {self.device} with dtype {self.dtype}.")

        self.counter = 0

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

                latents = encoded_bundle["latents"]
                os.environ["bs"] = str(latents.shape[0])
                os.environ["height"] = str(latents.shape[3] * 8)
                os.environ["width"] = str(latents.shape[4] * 8)    
                os.environ["frames"] = str((latents.shape[2] - 1) * 4 + 1)
                os.environ['sp'] = str(mpu.get_context_parallel_world_size())
                batch_size = latents.shape[0]
                timesteps = torch.tensor(0, device=torch.cuda.current_device()).long()
                # a. 加噪过程
                scheduler_sigmas = self.flow_scheduler.sigmas.clone()
                weights = compute_density_for_timestep_sampling(
                    weighting_scheme=self.flow_scheduler_config.flow_weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=self.flow_scheduler_config.flow_logit_mean,
                    logit_std=self.flow_scheduler_config.flow_logit_std,
                    mode_scale=self.flow_scheduler_config.flow_mode_scale,
                )
                indices = (
                    weights * (self.flow_scheduler.config.num_train_timesteps - 1)
                ).long()
                sigmas = scheduler_sigmas[indices].to(
                    device=torch.cuda.current_device()
                )
                timesteps = (sigmas * 1000.0).long()
                # broadcast_timesteps(timesteps)

                noise = torch.randn(latents.shape, device=torch.cuda.current_device())

                def expand_tensor_to_dims(tensor, ndim):
                    while len(tensor.shape) < ndim:
                        tensor = tensor.unsqueeze(-1)
                    return tensor

                sigmas = expand_tensor_to_dims(sigmas, ndim=latents.ndim)
                # print(f"rank {dist.get_rank()} latents shape is {latents.shape}")
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                tensors_to_cat = [noisy_model_input]

                # 按名字从“零件包”中取出并装配
                if "ref_images" in encoded_bundle:
                    tensors_to_cat.append(encoded_bundle["ref_latents"])
                if "first_ref_latents" in encoded_bundle:
                    tensors_to_cat.append(encoded_bundle["first_ref_latents"])
                    tensors_to_cat.append(encoded_bundle["first_ref_mask"])
                if "cn_latents" in encoded_bundle:
                    cn_latents = encoded_bundle["cn_latents"]
                    # 如果有对cn_latents的特殊处理（如guider），就在这里执行
                    tensors_to_cat.append(cn_latents)

                # 4. 最终拼接成模型输入
                model_input = torch.cat(tensors_to_cat, dim=1)
                latent_model_input = model_input.to(self.dtype)
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

                # c. 准备其他条件
                # 重要：对于 ref_images, cn_images 等，我们假设它们已经被外部的VAEPipeline处理成
                # conditional_latents 并放在了batch_dict中，以保持类之间的解耦。
                if "conditional_latents" in batch_dict:
                    # 此处应有拼接conditional_latents到noisy_model_input的逻辑
                    pass  # 示例：noisy_model_input = torch.cat([noisy_model_input, batch_dict["conditional_latents"]], dim=1)

                guidance = (
                    torch.tensor(
                        [1.0] * batch_size,
                        dtype=self.dtype,
                        device=torch.cuda.current_device(),
                    )
                    * 1000.0
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
            hidden_states=latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
            timestep=timesteps,  # [263]
            encoder_hidden_states=prompt_embeds,  # [1, 226, 4096]
            encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]] None
            pooled_projections=pooled_prompt_embeds,  # [1, 1, 768]
            guidance=guidance,  # []
            return_dict=False,
        )[0]
        global_timer.stop(forward_str)

        loss = None  # 初始化loss
        # 流水线最后阶段：计算损失
        if self.post_process:
            # 计算损失权重
            weights = compute_loss_weighting_for_sd3(
                weighting_scheme=self.flow_scheduler_config.get(
                    "flow_weighting_scheme", "sigma"
                ),
                sigmas=sigmas,
            )
            # Flow Matching的目标
            target = noise - latents
            # rank = dist.get_rank()
            # print(f"[rank {rank}] model_pred.shape: {model_pred.shape}")
            # print(f"[rank {rank}] target.shape: {target.shape}")
            # print(f"[rank {rank}] weights.shape: {weights.shape}")
            # 计算加权MSE损失
            loss = (
                weights.float() * (model_pred.float() - target.float()).pow(2)
            ).mean()

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
