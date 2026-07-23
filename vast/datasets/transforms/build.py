from vast.train.registry import Registry, build_module
from .prompt_transform import (
    PromptToClipEmbedding,
    PromptToTransformerEmbedding,
    PromptGenerator,
)
from .video_transform import (
    SampleImages, 
    GenerateRefImages, 
    GenerateFirstRefImage, 
)
from .formatting import PackInputs

TRANSFORMS = Registry()
TRANSFORMS.register_module(PromptToClipEmbedding)
TRANSFORMS.register_module(PromptToTransformerEmbedding)
TRANSFORMS.register_module(PromptGenerator)
TRANSFORMS.register_module(SampleImages)
TRANSFORMS.register_module(PackInputs)
TRANSFORMS.register_module(GenerateRefImages)
TRANSFORMS.register_module(GenerateFirstRefImage)


def build_transform(params_or_type, *args, **kwargs):
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
