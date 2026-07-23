import torch
import torch.nn as nn

from .CogVAEPipeline import CogVAEPipeline
from .CogDITPipeline import CogDiTPipeline
# from dit_pipeline import DiTPipeline
from megatron.core import mpu
from typing import Dict, Any
from contextlib import contextmanager
import os
import torch.distributed as dist


class TrainingWrapperModel(nn.Module):
    def __init__(self, cog_config, dit_model_config):
        """

        Args:
        """
        super().__init__()
        print("[CogTrainingWrapperModel] Initializing...")

        self.vae_pipeline = CogVAEPipeline(cog_config)

        self.dit_pipeline = CogDiTPipeline(cog_config, dit_model_config)
        self.dtype = torch.bfloat16
        print("[CogTrainingWrapperModel] Initialized successfully.")

    @contextmanager
    def null_context(self):
        yield

    def forward(self,
                batch_dict: Dict[str, Any],
                loss_mask: torch.Tensor = None,
                mode: str = 'full',
                precomputed_encoded_bundle: Dict[str, torch.Tensor] = None):
        """

        Args:
            precomputed_encoded_bundle (dict, optional):

        Returns:
        """

        if mode == 'vae_only':
            with torch.no_grad():
                encoded_bundle = self.vae_pipeline(batch_dict)
            return encoded_bundle

        if mode == 'full':
            with torch.no_grad():
                encoded_bundle = self.vae_pipeline(batch_dict)
        elif mode == 'dit_only':
            if precomputed_encoded_bundle is None:
                raise ValueError("In 'dit_only' mode, 'precomputed_encoded_bundle' must be provided.")
            encoded_bundle = precomputed_encoded_bundle
        else:
            raise ValueError(f"Unknown forward mode: '{mode}'. "
                             "Available modes are 'full', 'vae_only', 'dit_only'.")

        loss = self.dit_pipeline(batch_dict, encoded_bundle)

        return [loss]


    @property
    def transformer(self):
        return self.dit_pipeline.transformer

    @property
    def config(self):
        return self.dit_pipeline.transformer.config
    

    def get_trainable_params(self):
        print("[TrainingWrapperModel] Providing trainable parameters from DiTPipeline...")
        return self.dit_pipeline.parameters()

    # def state_dict_for_save_checkpoint(self, ...):
    #     return self.dit_pipeline.state_dict_for_save_checkpoint(...)
    
    # def load_state_dict(self, ...):
    #     self.dit_pipeline.load_state_dict(...)
    def set_input_tensor(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.transformer.set_input_tensor(input_tensor)
        pass

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)