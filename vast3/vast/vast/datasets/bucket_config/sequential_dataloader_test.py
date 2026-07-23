from bucket import Bucket
from read_cfg import get_cfg
from sampler import VariableVideoBatchSampler
from plot_cfg import *
import torch
# from vast.datasets.collators.default_collator import DefaultCollator
from vast.datasets.bucket_config.UniformDictTensorCollator import SimpleCollator
cfg = get_cfg('stage3.py')
# buc = Bucket(cfg.get('bucket_config',None))
from vast.datasets.datasets.build import build_dataset
data_config = cfg['dataloaders']['train']
dataset = build_dataset(data_config['dataset'])
sampler = VariableVideoBatchSampler(dataset,cfg.get('bucket_config',None),batch_size=8,
                                       num_replicas=1,rank=0)
# batch_sampler = BatchSampler(
#                 sampler, batch_size=1, drop_last=False
# )



# collator = data_config.get("collator", {})
collator = SimpleCollator()
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=data_config.get('num_workers',1),
        )  


for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}: {batch['images'].shape}")



    
# print(gb)
# del samplerins.seqlen2cnt[494527.0]
# print(sorted(samplerins.seqlen2cnt.items(), key = lambda item:item[1], reverse=True))
# plot_sequence_length_cdf(samplerins.seqlen2cnt,f'dataset_{samplerins.data_count}.png')

# plot_sequence_length_distribution(samplerins.seqlen2cnt,f'dataset_distribution_{samplerins.data_count}.png')