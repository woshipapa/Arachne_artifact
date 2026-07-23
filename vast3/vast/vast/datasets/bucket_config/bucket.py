from collections import OrderedDict

import numpy as np

# from opensora.utils.logger import log_message

from aspect import get_closest_ratio, get_resolution_with_aspect_ratio
from bucket_utils import map_target_fps


class Bucket:
    def __init__(self, bucket_config: dict[str, dict[int, tuple[float, int] | tuple[tuple[float, float], int]]]):
        """
        Args:
            bucket_config (dict): A dictionary containing the bucket configuration.
                The dictionary should be in the following format:
                {
                    "bucket_name": {
                        "time": (probability, batch_size),
                        "time": (probability, batch_size),
                        ...
                    },
                    ...
                }

                Or in the following format:
                {
                    "bucket_name": {
                        "time": ((probability, next_probability), batch_size),
                        "time": ((probability, next_probability), batch_size),
                        ...
                    },
                    ...
                }
                The bucket_name should be the name of the bucket, and the time should be the number of frames in the video.
                The probability should be a float between 0 and 1, and the batch_size should be an integer.
                If the probability is a tuple, the second value should be the probability to skip to the next time.
        """
        # ie. return (65536, {"16:9": (256, 144), "4:3": (192, 144), "1:1": (160, 160)})
        # print(f'{bucket_config}')
        aspect_ratios = {key: get_resolution_with_aspect_ratio(key) for key in bucket_config.keys()}
        # print(f'{aspect_ratios}')
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        # 按照num_pexels从大到小排序
        bucket_names = sorted(bucket_config.keys(), key=lambda x: aspect_ratios[x][0], reverse=True)

        #bucket_probs["256px"] = OrderedDict({
        #     129: 1.0,
        #     13: 1.0,
        #     9: 1.0,
        #     5: 1.0,
        #     1: 1.0,
        # })

        # bucket_bs["256px"] = OrderedDict({
        #     129: 3,
        #     13: 12,
        #     9: 12,
        #     5: 12,
        #     1: 45,
        # })

        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            # print(f'{bucket_probs[key]}')
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        self.hw_criteria = {k: aspect_ratios[k][0] for k in bucket_names}
        #{'1440px': {25: 25, 29: 29, 33: 33, 37: 37, 41: 41, 45: 45, 49: 49, 53: 53 ...}, '960px': }
        self.t_criteria = {k1: {k2: k2 for k2 in bucket_config[k1].keys()} for k1 in bucket_names}
        # print(f'{self.t_criteria}')
        # 这里相当于每个resolution下有多个时间帧，每个时间帧下又有多个aspect ratio

        # {'1440px': {25: {'2.39:1': (928, 2240), '2:1': (1024, 2032), '16:9': (1072, 1936), '1.85:1': (1056, 1968) ..
        self.ar_criteria = {
            k1: {k2: {k3: v3 for k3, v3 in aspect_ratios[k1][1].items()} for k2 in bucket_config[k1].keys()}
            for k1 in bucket_names
        }
        self.ar_seq_len =  {
            k1: {k2: {k3: v3[0]/16 * v3[1]/16 * ((k2-1)/4 + 1) for k3, v3 in aspect_ratios[k1][1].items()} for k2 in bucket_config[k1].keys()}
            for k1 in bucket_names
        }
        # print(f'ar_ {self.ar_seq_len}')
        bucket_id_cnt = num_bucket = 0
        bucket_id = dict()
        for k1, v1 in bucket_probs.items():
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                num_bucket += len(aspect_ratios[k1][1])
        # print(f'{bucket_id}')
        # bucket_id {'1440px': {241: 0, 237: 1, 233: 2, 229: 3, 225: 4, 221: 5,        
        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.num_bucket = num_bucket

        print(f"Number of buckets: {num_bucket}" )


        self.raw_count = {}

    # 调用此方法时，传入的参数就是实际存储视频数据的相关参数
    # 为每个video寻找他最合适的预定义的桶，包括resolution，时间帧数和aspect ratio
    def get_bucket_id(
        self,
        T: int,
        H: int,
        W: int,
        fps: float,
        path: str | None = None,
        seed: int | None = None,
        fps_max: int = 24,
    ) -> tuple[str, int, int] | None:
        approx = 0.8
        _, sampling_interval = map_target_fps(fps, fps_max)
        T = T // sampling_interval
        resolution = H * W
        rng = np.random.default_rng(seed)

        # Reference to probabilities and criteria for faster access
        bucket_probs = self.bucket_probs
        hw_criteria = self.hw_criteria
        ar_criteria = self.ar_criteria

        # Start searching for the appropriate bucket
        for hw_id, t_criteria in bucket_probs.items():
            # if resolution is too low, skip
            if resolution < hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    if rng.random() < t_criteria[1]:
                        return hw_id, 1, get_closest_ratio(H, W, ar_criteria[hw_id][1])
                continue

            # Look for suitable t_id for video
            for t_id, prob in t_criteria.items():
                if T >= t_id and t_id != 1:
                    # if prob is a tuple, use the second value as the threshold to skip
                    # to the next t_id
                    if isinstance(prob, tuple):
                        next_hw_prob, next_t_prob = prob
                        if next_t_prob >= 1 or rng.random() <= next_t_prob:
                            continue
                    else:
                        next_hw_prob = prob
                    if next_hw_prob >= 1 or rng.random() <= next_hw_prob:
                        ar_id = get_closest_ratio(H, W, ar_criteria[hw_id][t_id])
                        return hw_id, t_id, ar_id
                    else:
                        break

        return None

    def get_thw(self, bucket_idx: tuple[str, int, int]) -> tuple[int, int, int]:
        assert len(bucket_idx) == 3
        T = self.t_criteria[bucket_idx[0]][bucket_idx[1]]
        H, W = self.ar_criteria[bucket_idx[0]][bucket_idx[1]][bucket_idx[2]]
        return T, H, W

    def get_prob(self, bucket_idx: tuple[str, int]) -> float:
        return self.bucket_probs[bucket_idx[0]][bucket_idx[1]]

    def get_batch_size(self, bucket_idx: tuple[str, int]) -> int:
        return self.bucket_bs[bucket_idx[0]][bucket_idx[1]]

    def __len__(self) -> int:
        return self.num_bucket

    def update_raw_hw_count(self,
                         clip):
        self.raw_count[(clip.width,clip.height)] = self.raw_count.get((clip.width,clip.height),0) + 1
        





