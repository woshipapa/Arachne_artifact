import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from sampler import VariableVideoBatchSampler
from vast.datasets.collators.default_collator import DefaultCollator
from vast.datasets.bucket_config.UniformDictTensorCollator import SimpleCollator
from vast.datasets.datasets.build import build_dataset
from bucket import Bucket
from read_cfg import get_cfg


def main():
    # -------------------------------
    # 1. 初始化分布式环境
    # -------------------------------
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Initialized process group with world size {world_size}")

    # -------------------------------
    # 2. 加载配置和数据
    # -------------------------------
    cfg = get_cfg('stage3.py')
    data_config = cfg['dataloaders']['train']
    dataset = build_dataset(data_config['dataset'])

    # -------------------------------
    # 3. 初始化 Sampler（分布式 aware）
    # -------------------------------
    sampler = VariableVideoBatchSampler(
        dataset,
        cfg.get('bucket_config', None),
        batch_size=8,
        num_replicas=world_size,
        rank=rank
    )

    # -------------------------------
    # 4. 初始化 Collator
    # -------------------------------
    collator = SimpleCollator()

    # -------------------------------
    # 5. 创建 Dataloader（每卡独立）
    # -------------------------------
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=data_config.get('num_workers', 1),
        pin_memory=True
    )

    # -------------------------------
    # 6. 遍历数据
    # -------------------------------
    for batch_idx, batch in enumerate(dataloader):
        print(f"[Rank {rank}] Batch {batch_idx}: images.shape = {batch['clip_text_embed'].shape}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()