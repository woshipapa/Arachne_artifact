import copy
import random
import scipy
import numpy as np
import os
from vast.datasets.transforms import TRANSFORMS
from .prompt_transforms import PromptTransform
from .video_transforms import VideoTransform


@TRANSFORMS.register
class CogVideoXTransform:
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
            if "input_images" in data_dict:
                new_data_dict["images"] = data_dict["input_images"]

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
                # 实际上这里就应该报异常了
                new_data_dict["prompt_ids"] = data_dict["prompt_ids"]
                new_data_dict["prompt_masks"] = data_dict["prompt_masks"]

        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict


@TRANSFORMS.register
class CogVideoXTransform3D:
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

    def _convert2dposeto3d(self, pose, depthmap):
        video_height, video_width = depthmap.shape
        new_pose = {}
        for key in pose.keys():
            # bodies, faces, hands
            subsets = pose[key]["subset"]
            candidates = pose[key]["candidate"]
            valid_idx = np.all((candidates > 0) & (candidates < 1), axis=1)

            candidates[candidates < 0] = 0
            candidates[candidates >= 1] = 0.9999

            uv_coord = np.floor(
                candidates * np.array([video_width, video_height])
            ).astype(int)

            pad_depth = np.hstack(
                (candidates, depthmap[uv_coord[:, 1], uv_coord[:, 0]].reshape(-1, 1))
            )

            pad_depth[:, 0] = (pad_depth[:, 0] - 0.5) / 0.6
            pad_depth[:, 1] = (pad_depth[:, 1] - 0.5) / 0.6

            new_subsets = copy.deepcopy(subsets)
            for idx1 in range(subsets.shape[0]):
                for idx2 in range(subsets.shape[1]):
                    if (
                        subsets[idx1][idx2] != -1
                        and not valid_idx[subsets[idx1][idx2].astype(np.int32)]
                    ):
                        new_subsets[idx1][idx2] = -1
            new_pose[key]["subset"] = new_subsets
            new_pose[key]["candidate"] = candidates

        return new_pose

    def __call__(self, data_dict):
        data_dict = self.video_transform(data_dict)
        depth = scipy.sparse.load_npz(
            os.path.join(data_dict["root_dir"], data_dict["depth"])
        ).toarray()
        depth = (
            depth.reshape(data_dict["video_length"], data_dict["video_height"], -1)
            / 255.0
        )

        # convert poses to pose3d
        poses = data_dict["poses"]
        new_poses = [
            self._convert2dposeto3d(poses[i], depth[i]) for i in range(depth.shape[0])
        ]

        # 使用embed 则绕过
        if "prompt_embeds" not in data_dict.keys():
            if "prompt" in data_dict.keys():
                data_dict = self.prompt_transform(
                    data_dict, select_keys=["prompt"]
                )  # 由config.json得到
            else:
                data_dict = self.prompt_transform(
                    data_dict, select_keys=["prompt_cogvlm2", "prompt_panda"]
                )  # 由config.json得到

        if self.is_train:
            new_data_dict = {}
            if "input_images" in data_dict:
                new_data_dict["images"] = data_dict["input_images"]

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
                # 实际上这里就应该报异常了
                new_data_dict["prompt_ids"] = data_dict["prompt_ids"]
                new_data_dict["prompt_masks"] = data_dict["prompt_masks"]
        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict
