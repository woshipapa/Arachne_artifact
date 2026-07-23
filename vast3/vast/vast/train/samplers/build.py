from .aspect_ratio_sampler import AspectRatioSampler
from .bucket_batch_sampler import BucketBatchSampler
from .bucket_sampler import BucketSampler
from .default_sampler import DefaultSampler
from .parallel_batch_sampler import ParallelBatchSampler
from .special_sampler import SpecialDatasetSampler
from .two_dim_sampler import TwoDimSampler
# from vast.datasets.bucket_config.sampler import VariableVideoBatchSampler
# from vast.datasets.bucket_config.read_cfg import get_cfg
from ..registry import Registry, build_module

SAMPLERS = Registry()


SAMPLERS.register_module(AspectRatioSampler)
SAMPLERS.register_module(BucketBatchSampler)
SAMPLERS.register_module(BucketSampler)
SAMPLERS.register_module(DefaultSampler)
SAMPLERS.register_module(SpecialDatasetSampler)
SAMPLERS.register_module(TwoDimSampler)
# SAMPLERS.register_module(VariableVideoBatchSampler)

def build_sampler(params_or_type, *args, **kwargs):
    # print(params_or_type,type(params_or_type))
    # if params_or_type['type'] == 'VariableVideoBatchSampler':
    #     kwargs['bucket_config'] = get_cfg('stage3.py').get('bucket_config',None)
        # print(kwargs)
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)
