
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu
import torch.distributed as dist
from .WanModel import WanParams, WanModel
from .flow_match import FlowMatchScheduler
import os
class WanDiTPipeline(nn.Module):

    def __init__(self, wan_config: object, config: object):
        """

        Args:
        """
        super().__init__()
        print("[WanDiTPipeline] Initializing...")

        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()

        # if hasattr(hunyuan_config, "clip_params"):
        #     self.clip_encoder = FrozenCLIPEmbedder(**hunyuan_config.clip_params)
        # if hasattr(hunyuan_config, "t5_params"):
        #     self.t5_encoder = FrozenT5Encoder(**hunyuan_config.t5_params)

        wanConfig = WanParams()
        self.transformer = WanModel(wan_config = wanConfig, config = config)
        print("[DiTPipeline] WanModel created.")

        flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        flow_scheduler.set_timesteps(1000, training=True)
        self.flow_scheduler = flow_scheduler
        self.flow_scheduler_config = wan_config.get("scheduler", dict())

        if wan_config.get("lora", False):
            from peft import LoraConfig

            print("[DiTPipeline] Configuring LoRA...")
            self.transformer.requires_grad_(False)
            lora = wan_config.lora
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
        if self.pre_process:
            with torch.no_grad():
                latents = encoded_bundle["latents"]
                noise = torch.randn_like(latents) if "noise" not in batch_dict else batch_dict["noise"]

                timestep_id = torch.randint(0, self.flow_scheduler.num_train_timesteps, (1,))
                # index on the cpu
                timestep = self.flow_scheduler.timesteps[timestep_id].to(
                dtype=self.dtype, device=torch.cuda.current_device()
                    )
                

                def broadcast_timesteps(input: torch.Tensor):
                    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
                    if mpu.get_tensor_context_parallel_world_size() > 1:
                        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

                # broadcast_timesteps(timestep)
                # broadcast_timesteps(noise)
                training_target = self.flow_scheduler.training_target(latents, noise, timestep)

                noisy_latents = self.flow_scheduler.add_noise(latents, noise, timestep)
                prompt_embeds = batch_dict["prompt_embeds"].to(dtype=self.dtype)

        from my_utils import global_timer

        forward_str = f"forward_single{os.environ.get('NUM_WAN_LAYERS')}_bs_{os.environ.get('bs')}_f_{os.environ.get('frames')}_h_{os.environ.get('height')}_w_{os.environ.get('width')}_sp{mpu.get_context_parallel_world_size()}"
        global_timer.start(forward_str)
        output_tensor_list = self.transformer(x=noisy_latents, 
                               timestep=timestep, 
                                context = prompt_embeds,
                            #    clip_feature=batch_dict["clip_feature"],
                                clip_feature=None,
                            #    y=batch_dict["y"]
                                y=None
                               )
        global_timer.stop(forward_str)
        loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
        loss = loss * self.flow_scheduler.training_weight(timestep)
        self.counter += 1
        return loss
    

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.transformer.load_state_dict(state_dict, strict)

    def set_input_tensor(self, input_tensor):
        # self.transformer.set_input_tensor(input_tensor)
        pass