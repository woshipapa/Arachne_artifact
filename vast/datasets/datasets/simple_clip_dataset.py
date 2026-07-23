from typing import List
import numpy as np
from vast.datasets import image_utils
from .base_dataset import BaseDataset
from teleai_data_tool.schema.clip import Clip
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from teleai_data_tool.logger import logger
from collections import defaultdict
from tqdm import tqdm
from cattrs import structure


class SimpleClipDataset(BaseDataset):
    """

    [
        {
            "file_name": "video_001.mp4",
            "width": 1920,
            "height": 1080,
            "frame_count": 150,
            "fps": 30.0,
            "resolution": "1920x1080",
            "quality": "1080p"
        },
        ...
    ]

    """

    def __init__(
        self,
        data_path_list: List[str],
        transforms: callable,
        clip_data_root: str = "<DATA_ROOT>/output_clip_parallel/",
        clip_data_type: str = "file",
        filter_cfg: dict = dict(),
        data_weight_list: List = [],
        enable_bucket_index: bool = True,
    ) -> None:
        """

        Args:
        """
        self.data_path_list = data_path_list
        self.data_weight_list = data_weight_list
        self.enable_bucket_index = enable_bucket_index
        
        self.clip_data_root = clip_data_root
        self.clip_data_type = clip_data_type
        
        self.bucket_index_list = None
        
        super().__init__(
            ann_file="",
            serialize_data=False,
            test_mode=False,
            lazy_init=False,
            no_filter=True,
            max_refetch=1,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        self.file_client = FileClient()
        self.lmdb_client = LmdbClient()

    def load_data_list(self) -> List[Clip]:
        data_list = []
        print(f"Loading data from {len(self.data_path_list)} simplified JSON files...")
        for data_path in tqdm(self.data_path_list, desc="Parsing data files"):
            with open(data_path, 'r', encoding='utf-8') as f:
                clip_entries = json.load(f)
            
            for i,entry in enumerate(clip_entries):
                clip_dict = {
                    "id": i,
                    "file_path": entry["file_name"],
                    "length": entry["frame_count"],
                    "fps": entry["fps"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "filter_state": {
                        "aesthetic": None,
                        "laplacian": None,
                        "optical_flow": None,
                        "clearity": None,
                        "motion": None,
                        "video_training_suitability": None,
                        "area": entry["width"] * entry["height"]
                    },
                    "caption": {
                        "short_caption": "",
                        "dense_caption": "A high-quality video clip, featuring dynamic scenes and clear visuals.",
                    },
                    "meta": {
                        "quality": entry.get("quality", "unknown"),
                        "resolution": entry.get("resolution", "unknown")
                    },
                    "valid_range": [0, entry["frame_count"]]
                }
                
                clip = structure(clip_dict, Clip)
                
                clip.file_path = f"{self.clip_data_root}:{clip.file_path}"
                clip.meta["data_format"] = self.clip_data_type
                
                data_list.append(clip)
                
        print(f"Loaded a total of {len(data_list)} clips before filtering.")
        return data_list


    def filter_data(self) -> List[Clip]:
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

        for clip in self.data_list:
            frame_interval = max(1, round(clip.fps / dst_fps))
            min_num_frames = frame_interval * dst_num_frames
            setattr(clip, "frame_interval", frame_interval)
            
            if clip.height * clip.width < min_area:
                continue
            
            # aesthetic score
            if (clip.filter_state.aesthetic is None or clip.filter_state.aesthetic < aesthetic_th):
                continue

            # laplacian
            if (clip.filter_state.laplacian is not None and clip.filter_state.laplacian < laplacian_th):
                continue

            # optical_flow
            if clip.filter_state.optical_flow != -1.0:
                if (clip.filter_state.optical_flow is None or clip.filter_state.optical_flow < optical_flow_th):
                    continue
            
            # size
            if clip.filter_state.area < area_th:
                continue

            # length
            if clip.length < min_num_frames:
                continue

            # clearity
            if (clip.filter_state.clearity is not None and clip.filter_state.clearity < clearity_th):
                continue

            # motion
            if (clip.filter_state.motion is not None and clip.filter_state.motion < motion_th):
                continue

            # training_suitability
            if (clip.filter_state.video_training_suitability is not None and clip.filter_state.video_training_suitability < training_suitability_th):
                continue
        
            if self.enable_bucket_index:
                dst_width, dst_height = image_utils.get_image_size(
                    (clip.width, clip.height),
                    dst_size,
                    mode="area",
                    multiple=multiple,
                )
                video_info = f"{dst_width}__{dst_height}"
                if video_info not in shape_list:
                    shape_list.append(video_info)
                setattr(clip, "bucket_index", shape_list.index(video_info))
                shape_num_map[video_info] += 1
                setattr(clip, "video_info", (dst_width, dst_height))
            
            new_data_list.append(clip)

        if self.enable_bucket_index:
            invalid_bucket_id_list = []
            for video_info_str, count in shape_num_map.items():
                if count < bucket_size_th:
                    bucket_id_to_remove = shape_list.index(video_info_str)
                    invalid_bucket_id_list.append(bucket_id_to_remove)
            
            valid_data_list = [
                clip
                for clip in new_data_list
                if clip.bucket_index not in invalid_bucket_id_list
            ]
            
            bucket_index_list = defaultdict(list)
            for i, clip in enumerate(valid_data_list):
                bucket_index_list[clip.bucket_index].append(i)
            self.bucket_index_list = [np.array(item) for item in bucket_index_list.values()]
        else:
            valid_data_list = new_data_list

        print(
            f"Finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)}"
        )

        
        return valid_data_list

    def get_data_info(self, idx: int) -> dict:
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
        clip: Clip = super().get_data_info(idx)
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        dst_fps = self.filter_cfg.get("dst_fps", 24)

        frame_interval = max(1, round(clip.fps / dst_fps))
        min_num_frames = frame_interval * dst_num_frames
        setattr(clip, "frame_interval", frame_interval)
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

    def get_bucket_index_list(self):
        return self.bucket_index_list