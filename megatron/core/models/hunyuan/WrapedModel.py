import torch
import torch.nn as nn

# 假设您的两个Pipeline类在这里
from . import VAEPipeline
from .DITPipeline import DiTPipeline
# from dit_pipeline import DiTPipeline
from megatron.core import mpu
from typing import Dict, Any
from contextlib import contextmanager
import os
import torch.distributed as dist


class TrainingWrapperModel(nn.Module):
    """
    一个容器模型，用于将VAEPipeline和DiTPipeline封装成单一模块，
    以适配Megatron的训练框架。
    """
    def __init__(self, hunyuan_config, dit_model_config):
        """
        在容器内部，实例化所有需要的子模型。

        Args:
            hunyuan_config (object): 用于VAEPipeline的配置。
            dit_model_config (object): 用于DiTPipeline的Transformer部分的配置。
        """
        super().__init__()
        print("[TrainingWrapperModel] Initializing...")

        # 1. 实例化 VAE Pipeline
        # VAE是预训练且冻结的，它的参数不参与优化。
        self.vae_pipeline = VAEPipeline(hunyuan_config)

        # 2. 实例化 DiT Pipeline
        # DiT是我们需要训练的核心模型。
        # 注意: 这里的配置需要根据您的实际情况传入
        self.dit_pipeline = DiTPipeline(hunyuan_config, dit_model_config)
        self.dtype = torch.bfloat16
        print("[TrainingWrapperModel] Initialized successfully.")

    @contextmanager
    def null_context(self):
        """一个什么都不做的空上下文管理器，用于在不开启profile时作为占位符。"""
        yield

    def forward(self,
                batch_dict: Dict[str, Any],
                loss_mask: torch.Tensor = None,
                mode: str = 'full',
                precomputed_encoded_bundle: Dict[str, torch.Tensor] = None):
        """
        [核心修改] 定义了模型的多模式前向传播逻辑，并优雅地集成了可选的性能分析器。

        Args:
            batch_dict (dict): 从数据加载器传来的批次数据。
            loss_mask: Megatron可能传入的损失掩码。
            mode (str): 执行模式。可选值为 'full', 'vae_only', 'dit_only'。
            precomputed_encoded_bundle (dict, optional):
                在 mode='dit_only' 时使用，传入预先由VAE编码好的潜空间向量字典。

        Returns:
            - 如果 mode='full' 或 'dit_only'，返回包含损失的列表: [loss]。
            - 如果 mode='vae_only'，返回包含潜空间向量的字典: encoded_bundle。
        """

        # 模式二：仅执行VAE编码。此模式不涉及DiT，可以提前处理并返回。
        if mode == 'vae_only':
            with torch.no_grad():
                encoded_bundle = self.vae_pipeline(batch_dict)
            return encoded_bundle

        # 为 'full' 和 'dit_only' 模式准备 DiT 的输入
        if mode == 'full':
            with torch.no_grad():
                # 在 'full' 模式下，实时通过 VAE 计算 latents
                encoded_bundle = self.vae_pipeline(batch_dict)
        elif mode == 'dit_only':
            # 在 'dit_only' 模式下，使用预计算的 latents
            if precomputed_encoded_bundle is None:
                raise ValueError("In 'dit_only' mode, 'precomputed_encoded_bundle' must be provided.")
            encoded_bundle = precomputed_encoded_bundle
        else:
            # 处理未知的模式
            raise ValueError(f"Unknown forward mode: '{mode}'. "
                             "Available modes are 'full', 'vae_only', 'dit_only'.")

        loss = self.dit_pipeline(batch_dict, encoded_bundle)

        return [loss]


    @property
    def transformer(self):
        """
        [新增] 创建一个名为 'transformer' 的只读属性。
        当外部代码访问 model.transformer 时，它不会返回整个DiTPipeline，
        而是直接返回DiTPipeline内部真正的那个HunyuanVideoTransformer3DModel实例。
        """
        return self.dit_pipeline.transformer

    @property
    def config(self):
        """
        [新增] 同样地，为 'config' 属性创建一个代理。
        它直接返回最终Transformer模型的配置。
        """
        return self.dit_pipeline.transformer.config
    

    def get_trainable_params(self):
        """
        一个辅助方法，清晰地告诉优化器哪些参数需要训练。
        这在配置优化器时非常有用。
        """
        print("[TrainingWrapperModel] Providing trainable parameters from DiTPipeline...")
        # 我们只训练DiT Pipeline中的参数（特别是Transformer和可能的LoRA层）
        return self.dit_pipeline.parameters()

    # 如果需要，还可以将 state_dict 相关的逻辑也代理到这里
    # def state_dict_for_save_checkpoint(self, ...):
    #     # 只保存需要训练的部分
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