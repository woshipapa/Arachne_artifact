import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu
import torch.distributed as dist

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

# from your_model_library import FrozenCLIPEmbedder, FrozenT5Encoder, LoraConfig


def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(
            input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group()
        )


class DiTPipeline(nn.Module):

    def __init__(self, hunyuan_config: object, config: object):
        """

        Args:
        """
        super().__init__()
        print("[DiTPipeline] Initializing...")

        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()

        # if hasattr(hunyuan_config, "clip_params"):
        #     self.clip_encoder = FrozenCLIPEmbedder(**hunyuan_config.clip_params)
        # if hasattr(hunyuan_config, "t5_params"):
        #     self.t5_encoder = FrozenT5Encoder(**hunyuan_config.t5_params)

        hunyuanConfig = HunyuanParams()
        self.transformer = HunyuanVideoTransformer3DModel(hunyuanConfig, config)
        print("[DiTPipeline] HunyuanVideoTransformer3DModel created.")

        self.flow_scheduler = FlowMatchEulerDiscreteScheduler()
        self.flow_scheduler_config = hunyuan_config.get("scheduler", dict())

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
        # self.to(self.device)
        # print(f"[DiTPipeline] Model moved to {self.device} with dtype {self.dtype}.")

        self.counter = 0

    def forward(self, batch_dict: dict, encoded_bundle: dict) -> torch.Tensor:
        """

        Args:

        Returns:
        """
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

                if "ref_images" in encoded_bundle:
                    tensors_to_cat.append(encoded_bundle["ref_latents"])
                if "first_ref_latents" in encoded_bundle:
                    tensors_to_cat.append(encoded_bundle["first_ref_latents"])
                    tensors_to_cat.append(encoded_bundle["first_ref_mask"])
                if "cn_latents" in encoded_bundle:
                    cn_latents = encoded_bundle["cn_latents"]
                    tensors_to_cat.append(cn_latents)

                model_input = torch.cat(tensors_to_cat, dim=1)
                latent_model_input = model_input.to(self.dtype)
                prompt_embeds = batch_dict["prompt_embeds"].to(dtype=self.dtype)
                pooled_prompt_embeds = batch_dict["clip_text_embed"].to(
                    dtype=self.dtype
                )
                prompt_masks = (
                    batch_dict["prompt_masks"].to(dtype=self.dtype)
                    if "prompt_masks" in batch_dict
                    else None
                )

                if "conditional_latents" in batch_dict:
                    pass

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

        loss = None
        if self.post_process:
            weights = compute_loss_weighting_for_sd3(
                weighting_scheme=self.flow_scheduler_config.get(
                    "flow_weighting_scheme", "sigma"
                ),
                sigmas=sigmas,
            )
            target = noise - latents
            # rank = dist.get_rank()
            # print(f"[rank {rank}] model_pred.shape: {model_pred.shape}")
            # print(f"[rank {rank}] target.shape: {target.shape}")
            # print(f"[rank {rank}] weights.shape: {weights.shape}")
            loss = (
                weights.float() * (model_pred.float() - target.float()).pow(2)
            ).mean()

        self.counter += 1
        return loss


    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.transformer.load_state_dict(state_dict, strict)

    def set_input_tensor(self, input_tensor):
        # self.transformer.set_input_tensor(input_tensor)
        pass
