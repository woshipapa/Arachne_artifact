import copy
import math
import random
import json
import scipy
import numpy as np
import torch
import os
from vast.datasets import video_utils


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "image_head",
            "image_tail",
            "image_head_tail",
            "image_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "quarter_random",
            "interpolation",
            "random",
            "identity",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        self.mask_ratios = mask_ratios

    def get_mask(self, num_frames):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break
        condition_frames_max = max(1, num_frames // 4)
        mask = torch.ones(num_frames, dtype=torch.bool)
        if num_frames <= 1:
            return mask
        if mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "interpolation":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1
        return mask


class GenerateRefImages:
    def __init__(self, mask_cfg=dict()):
        self.mask_generator = MaskGenerator(mask_cfg)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)[:, None, None, None]
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images
        return data_dict


class GenerateFirstRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict


class GenerateRepeatedFirstImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict


class GeneratePoseControlImages:
    def __init__(self):
        pass


class SampleImages:
    def __init__(
        self,
        num_frames=1,
    ):
        self.num_frames = num_frames

    def __call__(self, data_dict):
        video = data_dict["video"]
        # print(f'video is {video}')
        sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
        # print(f"rank {torch.distributed.get_rank()}  {len(sample_indexes)}")
        images = video_utils.sample_video(video, sample_indexes, method=2)
        # T,C,H,W
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        # print(f'rank {torch.distributed.get_rank()} images is {images.shape}')
        data_dict["images"] = images
        # print(f'--------------------------data_dict is {data_dict}')
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]
        
        # frame_interval = data_dict["frame_interval"]
        frame_interval = 1
        sample_length = (num_frames - 1) * frame_interval + 1
        # print(f"rank {torch.distributed.get_rank()} vlength = {video_length}, num_frames = {num_frames}, sample_length = {sample_length}")
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes
