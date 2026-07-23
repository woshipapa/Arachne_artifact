from .aspect_ratio_sampler import AspectRatioSampler
from .bucket_batch_sampler import BucketBatchSampler
from .bucket_sampler import BucketSampler
from .default_sampler import DefaultSampler
from .parallel_batch_sampler import ParallelBatchSampler
from .special_sampler import SpecialDatasetSampler
from .two_dim_sampler import TwoDimSampler
from ..registry import Registry, build_module
from vast.datasets.bucket_config.bucket_sampler import VariableVideoBatchSampler
SAMPLERS = Registry()


SAMPLERS.register_module(AspectRatioSampler)
SAMPLERS.register_module(BucketBatchSampler)
SAMPLERS.register_module(BucketSampler)
SAMPLERS.register_module(DefaultSampler)
SAMPLERS.register_module(SpecialDatasetSampler)
SAMPLERS.register_module(TwoDimSampler)


def build_sampler(params_or_type, *args, **kwargs):
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)
