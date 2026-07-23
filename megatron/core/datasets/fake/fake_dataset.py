from typing import List
import numpy as np 
from .image_utils import *
from .base_dataset import BaseDataset
from collections import defaultdict
import torch
import random
import string


class FakeDataset(BaseDataset):
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
            max_refetch=10,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        # self.file_client = FileClient()
        # self.lmdb_client = LmdbClient()
        print("---------------------------------------------------------------- Fake Dataset --------------------------------------------------------")

    def load_data_list(self) -> List[dict]:
        data_list = [{} for _ in range(10000)]
        
        # for data_path in tqdm(self.data_path_list):
        #     with open(data_path) as f:
        #         dataset = json.load(f)
        #     for clip in dataset["clips"]:
        #         clip = structure(clip, Clip)
        #         clip.file_path = f"{dataset['clip_data_root']}:{clip.file_path}"
        #         clip.meta["data_format"] = dataset["clip_data_type"]
        #         # print(f"clip: {clip}")
        #         data_list.append(clip)

        # exit(0)

        # dataset_size  = self.filter_cfg.get("fake_dataset_size", 1000)
        # dst_size = self.filter_cfg.get("dst_size", (720, 480))
        # dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        # for i in range(10000):
        #     random_data = {}
            # random_data["prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
            # random_data["images"] = torch.randn((dst_num_frames, 3, dst_size[1], dst_size[0]))
            # random_data["first_ref_image"] = torch.randn((1, 3, dst_size[1], dst_size[0]))
            # random_data["prompt_embeds"] = torch.randn(120, 4096)
            # random_data["clip_text_embed"] = torch.randn(768)
            # data_list.append(random_data)
        # print("\n--------------------------------------> data info:\n", data_list[0])
        # print("<--------------------------------------\n")
        return data_list


    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        # self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


    def get_bucket_index_list(self):
        return self.bucket_index_list

    def filter_data(self):
        dst_size = self.filter_cfg.get("dst_size", (720, 480))
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        # print(f"dst_num_frames: {dst_num_frames}")
        dst_fps = self.filter_cfg.get("dst_fps", 24)
        multiple = self.filter_cfg.get("multiple", 16)
        min_area = self.filter_cfg.get("min_area", dst_size[0] * dst_size[1])
        optical_flow_th = self.filter_cfg.get("optical_flow_th", 2)
        aesthetic_th = self.filter_cfg.get("aesthetic_th", 4)
        bucket_size_th = self.filter_cfg.get("bucket_size_th", 4)
        motion_th = self.filter_cfg.get("motion_th", 0) 
        clearity_th = self.filter_cfg.get("clearity_th", 0.8) 
        laplacian_th = self.filter_cfg.get("laplacian_th", 0)
        training_suitability_th = self.filter_cfg.get("training_suitability_th", 3.7) 
        area_th = self.filter_cfg.get("area_th", 0)
        # fileter tag 
        too_small = 0
        too_short = 0
        motion_mismatch = 0
        aes_mismatch = 0
        motion_mismatch = 0
        clearity_mismatch = 0
        motion_mismatch = 0
        suitability_mismatch = 0
        buckets_mismatch = 0
        new_data_list = []
        shape_list = []
        shape_num_map = defaultdict(int)
        for clip in self.data_list:
            frame_interval = max(1, round(clip.fps / dst_fps))
            min_num_frames = frame_interval * dst_num_frames
            setattr(clip, "frame_interval", frame_interval)

            # length
            if clip.length < min_num_frames:
                too_short += 1
                continue
            # size
            if clip.height * clip.width < min_area:
                too_small += 1
                continue
            # aesthetic
            if (
                clip.filter_state.aesthetic is None
                or clip.filter_state.aesthetic < aesthetic_th
            ):
                aes_mismatch += 1
                continue

            if (
                clip.filter_state.laplacian is not None
                and clip.filter_state.laplacian < laplacian_th
            ):
                clearity_mismatch += 1
                continue

            # optical_flow
            if clip.filter_state.optical_flow != -1.0:
                if (
                    clip.filter_state.optical_flow is None
                    or clip.filter_state.optical_flow < optical_flow_th
                ):
                    motion_mismatch += 1
                    continue
            
            # size
            if clip.filter_state.area < area_th:
                too_small += 1
                continue

            # clearity
            if (
                clip.filter_state.clearity is not None
                and clip.filter_state.clearity < clearity_th
            ):
                clearity_mismatch += 1
                continue

            # motion
            if (
                clip.filter_state.motion is not None
                and clip.filter_state.motion < motion_th
            ):
                motion_mismatch += 1
                continue

            # training_suitability
            if (
                clip.filter_state.video_training_suitability is not None
                and clip.filter_state.video_training_suitability < training_suitability_th
            ):
                suitability_mismatch += 1
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
            for k, v in shape_num_map.items():
                if v < bucket_size_th:
                    buckets_mismatch += v
                    invalid_bucket_id_list.append(k)
            valid_data_list = [
                clip
                for clip in new_data_list
                if shape_list[clip.bucket_index] not in invalid_bucket_id_list
            ]
            bucket_index_list = defaultdict(list)
            for i, clip in enumerate(valid_data_list):
                bucket_index_list[clip.bucket_index].append(i)
            self.bucket_index_list = [
                np.array(item) for item in bucket_index_list.values()
            ]
        else:
            valid_data_list = new_data_list
        # logger.info(
        #     f"finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)} \n"
        #     f"too short data {too_short} \n"
        #     f"too small data {too_small} \n"
        #     f"motion mismatch data {motion_mismatch} \n"
        #     f"aesthetic mismatch data {aes_mismatch} \n"
        #     f"clearity score mismatch data {clearity_mismatch} \n"
        #     f"suitability score mismatch data {suitability_mismatch} \n"
        #     f"buckets mismatch data {buckets_mismatch} \n"
        #     f"bucket shape: {shape_num_map}" 
        # )
        return valid_data_list

    # def get_data_info(self, idx):
    #     clip: Clip = super().get_data_info(idx)
    #     data_dict = dict()
    #     if clip.meta["data_format"] == "lmdb":
    #         video = self.lmdb_client.get(clip.file_path)
    #     elif clip.meta["data_format"] == "file":
    #         video = self.file_client.get(clip.file_path)
    #     data_dict["clip_info"] = clip
    #     data_dict["video"] = video
    #     data_dict["video_info"] = clip.video_info
    #     data_dict["video_length"] = clip.length
    #     data_dict["video_height"] = clip.height
    #     data_dict["video_width"] = clip.width
    #     data_dict["video_valid_range"] = clip.valid_range
    #     data_dict["fps"] = clip.fps
    #     data_dict["frame_interval"] = clip.frame_interval
    #     return data_dict


    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        # import torch.distributed as dist 
        # idx_pool = [801, 805, 809, 813]
        # idx = idx_pool[idx%len(idx_pool)]
        # print(f"base dataset idx at getitem, rank {dist.get_rank()}, data {idx}")

        #-----------------------
        # if not self._fully_initialized:
        #     logger.info(
        #         "Please call `full_init()` method manually to accelerate " "the speed."
        #     )
        #     self.full_init()

        # if self.test_mode:
        #     data = self.prepare_data(idx)
        #     if data is None:
        #         raise Exception(
        #             "Test time pipline should not get `None` " "data_sample"
        #         )
        #     # print("-----------------------------> data: ", data)
        #     # exit(0)
        #     return data

        # for _ in range(self.max_refetch + 1):
        #     data = self.prepare_data(idx)
        #     # Broken images or random augmentations may cause the returned data
        #     # to be None
        #     if data is None:
        #         idx = self._rand_another()
        #         continue
        #     print("================================== get fake item ==================================")
        #     print("\n-------------------------------------------> data.shape: ")
        #     random_data = {}
        #     dst_size = self.filter_cfg.get("dst_size", (720, 480))
        #     dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)

            
        #     random_data["prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
        #     random_data["images"] = torch.randn((dst_num_frames, 3, dst_size[1], dst_size[0]))
        #     random_data["first_ref_image"] = torch.randn((1, 3, dst_size[1], dst_size[0]))
        #     random_data["prompt_embeds"] = torch.randn(120, 4096)
        #     random_data["clip_text_embed"] = torch.randn(768)

        #     print("-------------------------------------------       ")
        #     for key, value in data.items():
        #         if isinstance(value, torch.Tensor):
        #             print(f"data         Key: {key}, Value shape: {value.shape}")
        #         else:
        #             print(f"data          Key: {key}, Value.size: {len(value)}")
        #     print("\n")

        #     for key, value in random_data.items():
        #         if isinstance(value, torch.Tensor):
        #             print(f"random_data         Key: {key}, Value shape: {value.shape}")
        #         else:
        #             print(f"random_data          Key: {key}, Value.size: {len(value)}")
        #     print("\n")
        #     print("-------------------------------------------       ")
        #     # exit(0)
        #     return random_data

        dst_size = self.filter_cfg.get("dst_size", (720, 480))
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        from megatron.training import get_args
        args = get_args()
        if args.num_frames:
            dst_num_frames = args.num_frames
        if args.video_resolution:
            dst_size = tuple(args.video_resolution)
        random_data = {}
        random_data["prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["images"] = torch.randn((dst_num_frames, 3, dst_size[1], dst_size[0]))
        random_data["first_ref_image"] = torch.randn((1, 3, dst_size[1], dst_size[0]))
        random_data["prompt_embeds"] = torch.randn(120, 4096)
        random_data["clip_text_embed"] = torch.randn(768)

        # print("-------------------------------------------       ")
        # for key, value in random_data.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"random_data         Key: {key}, Value shape: {value.shape}")
        #     else:
        #         print(f"random_data          Key: {key}, Value.size: {len(value)}")
        # print("\n")
        # print("-------------------------------------------       ")
        return random_data
