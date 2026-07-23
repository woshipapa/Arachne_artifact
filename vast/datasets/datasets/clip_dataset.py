from typing import List
import numpy as np
from vast.datasets import image_utils
from .base_dataset import BaseDataset
from teleai_data_tool.schema.dataset import ClipsDataset as _ClipsDataset
from teleai_data_tool.schema.clip import Clip
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from teleai_data_tool.logger import logger
from collections import defaultdict
from tqdm import tqdm
from cattrs import structure


class ClipDataset(BaseDataset):
    def __init__(
        self,
        data_path_list,
        transforms,
        filter_cfg=dict(),
        data_weight_list=[],
        enable_bucket_index=True,
    ) -> None:
        self.data_path_list = data_path_list
        self.data_weight_list = data_weight_list
        self.enable_bucket_index = enable_bucket_index
        self.bucket_index_list = None
        super().__init__(
            ann_file="",
            serialize_data=False,
            test_mode=False,
            lazy_init=False,
            max_refetch=1,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        self.file_client = FileClient()
        self.lmdb_client = LmdbClient()

    def load_data_list(self) -> List[dict]:
        data_list = []
        for data_path in tqdm(self.data_path_list):
            with open(data_path) as f:
                dataset = json.load(f)
            # print(f'{dataset["clip_data_root"]}')    
            for clip in dataset["clips"]:
                # print(f'clip : {clip}')
                clip = structure(clip, Clip)
                clip.file_path = f"{dataset['clip_data_root']}:{clip.file_path}"
                clip.meta["data_format"] = dataset["clip_data_type"]
                data_list.append(clip)
        return data_list

    def get_bucket_index_list(self):
        return self.bucket_index_list

    def filter_data(self):
        dst_size = self.filter_cfg.get("dst_size", (720, 480))
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        dst_fps = self.filter_cfg.get("dst_fps", 24)
        multiple = self.filter_cfg.get("multiple", 16)
        min_area = self.filter_cfg.get("min_area", dst_size[0] * dst_size[1])
        optical_flow_th = self.filter_cfg.get("optical_flow_th", 2)
        aesthetic_th = self.filter_cfg.get("aesthetic_th", 4)
        bucket_size_th = self.filter_cfg.get("bucket_size_th", 4)
        motion_th = self.filter_cfg.get("motion_th", 0) 
        clearity_th = self.filter_cfg.get("clearity_th", 0.8) 
        laplacian_th = self.filter_cfg.get("laplacian", 0)
        training_suitability_th = self.filter_cfg.get("training_suitability_th", 3.7) 
        area_th = self.filter_cfg.get("area_th", 1280 * 720)
        new_data_list = []
        shape_list = []
        shape_num_map = defaultdict(int)
        # hw_set = set()
        # raw_count = {}
        # new_hw_set = set()
        # new_hw_count = {}
        # length_set = set()
        for clip in self.data_list:
            frame_interval = max(1, round(clip.fps / dst_fps))
            min_num_frames = frame_interval * dst_num_frames
            setattr(clip, "frame_interval", frame_interval)
            # length_set.add(clip.length)
            if clip.height * clip.width < min_area:
                continue
            # aesthetic
            if (
                clip.filter_state.aesthetic is None
                or clip.filter_state.aesthetic < aesthetic_th
            ):
                continue

            if (
                clip.filter_state.laplacian is not None
                and clip.filter_state.laplacian < laplacian_th
            ):
                continue

            # optical_flow
            if clip.filter_state.optical_flow != -1.0:
                if (
                    clip.filter_state.optical_flow is None
                    or clip.filter_state.optical_flow < optical_flow_th
                ):
                    continue
            
            # size
            if clip.filter_state.area < area_th:
                continue

            # length
            if clip.length < min_num_frames:
                continue

            # clearity
            if (
                clip.filter_state.clearity is not None
                and clip.filter_state.clearity < clearity_th
            ):
                continue

            # motion
            if (
                clip.filter_state.motion is not None
                and clip.filter_state.motion < motion_th
            ):
                continue

            # training_suitability
            if (
                clip.filter_state.video_training_suitability is not None
                and clip.filter_state.video_training_suitability < training_suitability_th
            ):
                continue
        
            if self.enable_bucket_index:
                dst_width, dst_height = image_utils.get_image_size(
                    (clip.width, clip.height),
                    dst_size,
                    mode="area",
                    multiple=multiple,
                )
                # hw_set.add((clip.width,clip.height))
                # raw_count[(clip.width,clip.height)] = raw_count.get((clip.width,clip.height),0) + 1
                # new_hw_set.add((dst_width,dst_height))
                # print(f'c : ({clip.width},{clip.height})---------->({dst_width},{dst_height})')
                video_info = f"{dst_width}__{dst_height}"
                if video_info not in shape_list:
                    shape_list.append(video_info)
                setattr(clip, "bucket_index", shape_list.index(video_info))
                shape_num_map[video_info] += 1
                setattr(clip, "video_info", (dst_width, dst_height))
            new_data_list.append(clip)
        if self.enable_bucket_index:
            invalid_bucket_id_list = []
            for k, v in shape_num_map.items():
                if v < bucket_size_th:
                    invalid_bucket_id_list.append(k)
            valid_data_list = [
                clip
                for clip in new_data_list
                if clip.bucket_index not in invalid_bucket_id_list
            ]
            bucket_index_list = defaultdict(list)
            for i, clip in enumerate(valid_data_list):
                bucket_index_list[clip.bucket_index].append(i)
            self.bucket_index_list = [
                np.array(item) for item in bucket_index_list.values()
            ]
        else:
            valid_data_list = new_data_list


        # new_area_list = [int(w*h) for (w,h) in new_hw_set]
        # new_area_list = sorted(new_area_list)    
        logger.info(
            f"finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)}"
        )


        # logger.info(
        #     f"frame length is {length_set}\n clip_hw_set is {hw_set}\n new_hw_set = {new_hw_set}\n "
        # )

        # for (width, height), count in raw_count.items():
        #     logger.info(f"Resolution: {width}x{height}, Count: {count}")

        return valid_data_list

    def get_data_info(self, idx):
        data_dict = dict()
        video_info = None
        # print(f'clip_dataset get_data_info idx {idx}, type is {type(idx)}--->')
        import torch.distributed as dist
        # # print(f'[Rank {dist.get_rank()}][Clip_dataset]: fetching {idx}')
        if isinstance(idx, str):
            # print(f'[Clip_dataset]: fetching {idx}')

            
            idx, num_frames, height, width = [int(val) for val in idx.split("-")]    
            video_info = (height, width)
            self.pipeline.set_sample_num_frames(num_frames=num_frames)
            data_dict["num_frames"] = num_frames

        # print(f'[Rank {dist.get_rank()}] after str process clip idx = {idx} video_info is {video_info}---->')    
        clip: Clip = super().get_data_info(idx)

        if clip.meta["data_format"] == "lmdb":
            video = self.lmdb_client.get(clip.file_path)
        elif clip.meta["data_format"] == "file":
            video = self.file_client.get(clip.file_path)
        data_dict["clip_info"] = clip
        data_dict["video"] = video
        data_dict["video_info"] = clip.video_info if video_info is None else video_info
        data_dict["video_length"] = clip.length
        data_dict["video_height"] = clip.height
        data_dict["video_width"] = clip.width
        data_dict["video_valid_range"] = clip.valid_range
        data_dict["fps"] = clip.fps
        data_dict["frame_interval"] = clip.frame_interval
        return data_dict
