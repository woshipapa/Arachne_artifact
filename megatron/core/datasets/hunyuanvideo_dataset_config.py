from dataclasses import dataclass
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

#TODO: inherit GPtDatasetConfig. omit for develop
@dataclass
class HunyuanVideoDatasetConfig():
    train_ds_config: dict = None
    eval_ds_config: dict = None