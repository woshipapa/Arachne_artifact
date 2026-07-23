import sys
sys.path.append("/root/bucket_vast")
from bucket import Bucket
from read_cfg import get_cfg
from vast.datasets.bucket_config.bucket_sampler import VariableVideoBatchSampler
from plot_cfg import *
cfg = get_cfg('stage3.py')
# buc = Bucket(cfg.get('bucket_config',None))


# sys.path.append("/root/deepspeed_baseline/vast3/vast")
from vast.datasets.datasets.build import build_dataset
data_config = cfg['dataloaders']['train']
dataset = build_dataset(data_config['dataset'])
samplerins = VariableVideoBatchSampler(dataset,cfg.get('bucket_config',None), 
                                       data_parallel_size=4,
                                       data_parallel_rank=0)
# print(buc)
it = iter(samplerins)
for _ in range(10000):
    print(next(it))
# print(gb)
# del samplerins.seqlen2cnt[494527.0]
print(sorted(samplerins.seqlen2cnt.items(), key = lambda item:item[1], reverse=True))
plot_sequence_length_cdf(samplerins.seqlen2cnt,f'dataset_{samplerins.total_samples}.png')

plot_sequence_length_distribution(samplerins.seqlen2cnt,f'dataset_distribution_{samplerins.total_samples}.png')