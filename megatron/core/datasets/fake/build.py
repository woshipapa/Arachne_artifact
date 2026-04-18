from .registry import Registry, build_module
from .fake_dataset import FakeDataset

DATASETS = Registry()
DATASETS.register_module(FakeDataset)


def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)