import copy
import random
import scipy
import numpy as np
import os
from .prompt_transforms import TRANSFORMS, PromptTransform
from .video_transforms import VideoTransform


@TRANSFORMS.register
class VASTBASETransform:
    def __init__(
        self,
        size_method=1,
        dst_size=None,
        flip=False,
        num_frames=1,
        stride=1,
        cn_mode=None,
        mask_ratios=dict(),
        prompt_cfg=None,
        is_train=False,
        with_ref_images=True,
    ):
        self.is_train = is_train
        self.with_ref_images = with_ref_images
        self.cn_mode = cn_mode
        self.video_transform = VideoTransform(
            size_method=size_method,
            dst_size=dst_size,
            flip=flip,
            num_frames=num_frames,
            stride=stride,
            cn_mode=cn_mode,
            mask_ratios=mask_ratios,
        )
        self.prompt_transform = PromptTransform(is_train=is_train, **prompt_cfg)

    def __call__(self, data_dict):
        data_dict = self.prompt_transform.preprocess(
            data_dict, num_frames=self.video_transform.num_frames
        )
        data_dict = self.video_transform(data_dict)
        data_dict = self.prompt_transform(data_dict)
   
        if self.is_train:
            new_data_dict = {}
            if 'input_images' in data_dict:
                new_data_dict['images'] = data_dict['input_images']

            if "clip_text_embed" in data_dict:
                new_data_dict["clip_text_embed"] = data_dict["clip_text_embed"]

            if "prompt_masks" in data_dict:
                new_data_dict["prompt_masks"] = data_dict["prompt_masks"]

            if self.with_ref_images and "input_ref_images" in data_dict:
                new_data_dict["ref_images"] = data_dict["input_ref_images"]

            if self.cn_mode is not None and "input_cn_images" in data_dict:
                new_data_dict["cn_images"] = data_dict["input_cn_images"]
            if "prompt_embeds" in data_dict:
                # 考虑有多个的情况
                new_data_dict["prompt_embeds"] = (
                    random.choice(data_dict["prompt_embeds"])
                    if isinstance(data_dict["prompt_embeds"], list)
                    else data_dict["prompt_embeds"]
                )
        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict
