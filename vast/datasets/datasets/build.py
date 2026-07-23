from vast.train.registry import Registry, build_module
from .dataset import ConcatDataset
from .clip_dataset import ClipDataset
from .simple_clip_dataset import SimpleClipDataset
DATASETS = Registry()
DATASETS.register_module(ConcatDataset)
DATASETS.register_module(ClipDataset)
DATASETS.register_module(SimpleClipDataset)

def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)
