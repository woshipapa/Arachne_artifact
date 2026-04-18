
import numpy as np
from typing import Iterator
from torch.utils.data import Dataset, DistributedSampler
import torch.distributed as dist
import torch

from vast.datasets.bucket_config.bucket_utils import gather_micro_batches_across_gpus
from vast.datasets.bucket_config.bucket import Bucket
from megatron.core import mpu
from collections import defaultdict, OrderedDict



class VariableVideoBatchSampler(DistributedSampler):

    def __init__(self,
                 dataset,
                 bucket_config: dict,
                #  batch_size: int,
                data_parallel_size: int | None = None,
                data_parallel_rank: int | None = None,
                shuffle: bool = True,
                seed: int = 0,
                drop_last: bool = False,) -> None:
        super().__init__(
            dataset=dataset, num_replicas=data_parallel_size, rank=data_parallel_rank, shuffle=shuffle, seed=seed, drop_last=drop_last
            )
        # from config defined bucket
        self.bucket = Bucket(bucket_config)

        # dataset
        # self.batch_size = batch_size
        self.dataset = dataset
        self.total_samples = len(self.dataset)
        # pratical data divided into buckets
        self.runtime_bucket = {}
        self.sorted_bucket_map = {} # sort by count
        self.seqlen2cnt = {}
        # train sample dict 
        self._cached_bucket_sample_dict = None
        self._cached_num_total_batch = None
        self.last_micro_batch_access_index = 0

        # self.seed = seed
        # self.shuffle = shuffle
        # self.num_replicas = num_replicas
        # self.drop_last = drop_last
        # self.rank = rank
        # self.seed = seed
        # self.epoch = 20


    def __iter__(self) -> Iterator[list[int]]:
        bucket_sample_dict, _ = self.group_by_bucket()
        # del bucket_sample_dict[('1440px', 241, '16:9')]
        # print(bucket_sample_dict)
        from my_utils import get_logger
        Logger = get_logger()
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()
        # print(bucket_sample_dict)
        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            Logger.info(f"[Rank {torch.distributed.get_rank()}] bucket_id = {bucket_id}, len(data_list) = {len(data_list)}")
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            # 把桶内的数据打乱
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            # 记录了每一个桶需要多少个micro-batch
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # print(f'bucket_micro_batch_count = {bucket_micro_batch_count}')

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by 
        # dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
  
        Logger.info(f'len bucket_id_access_order is {len(bucket_id_access_order)}')
        # print(f'len bucket_id_access_order is {len(bucket_id_access_order)}')        
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas
        # print(bucket_id_access_order)
        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas : (i + 1) * self.num_replicas]
            # print(f'iter {i}  start_iter_idx = {start_iter_idx}, num_iters = {num_iters}  bucket_access_list is {bucket_access_list}')
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # print(f'rank = {dist.get_rank()}')
            # debug_list = gather_micro_batches_across_gpus(cur_micro_batch, mpu.get_tensor_model_parallel_group())
            # # if mpu.get_data_parallel_rank() == 0:
            # print(f'debug_list = {debug_list}')
            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            # print(f'[Rank {dist.get_rank()}] iter {i} sampler __iter__ mb is {cur_micro_batch}')
            Logger.info(f'[Rank {dist.get_rank()}] iter {i} sampler __iter__ mb is {cur_micro_batch}')
            yield cur_micro_batch
        
        self.reset()
  
    def reset(self):
        self.last_micro_batch_access_index = 0

    def group_by_bucket(self) -> dict:

        data_list = self.dataset.data_list
        # print(data_list[0].width,len(data_list))  # pass
        bucket_ids = []

        num_skipped_samples = 0

        # if dist.get_rank() == 0:
        for clip in data_list:
                self.bucket.update_raw_hw_count(clip)
                valid_range = clip.valid_range
                valid_range = [int(idx) for idx in valid_range]
                key = self.bucket.get_bucket_id(valid_range[1]-valid_range[0],clip.height,clip.width,clip.fps)
                if key is None:
                    num_skipped_samples += 1
                else:
                    
                    bucket_ids.append(key)
                # self.runtime_bucket[key] = self.runtime_bucket.get(key,0) + 1

        # dist.barrier()
        # bucket_ids = sync_object_across_devices(bucket_ids)
        # dist.barrier()
        if num_skipped_samples > 0 and dist.get_rank() == 0: # 只在主进程打印一次
            print(f"[VariableVideoBatchSampler Warning] Skipped {num_skipped_samples} out of {len(data_list)} samples because they did not fit into any defined bucket.")
        bucket_sample_dict = defaultdict(list)
        # value_to_indices = defaultdict(list)
        for idx, val in enumerate(bucket_ids):
             bucket_sample_dict[val].append(idx)
        # print(bucket_sample_dict)     

        self._cached_bucket_sample_dict = bucket_sample_dict    

        num_total_batch = self.print_bucket_info(bucket_sample_dict)
        self._cached_num_total_batch = num_total_batch
        return bucket_sample_dict, num_total_batch
    

        # for k,v in self.runtime_bucket.items():
        #     sl = self.bucket.ar_seq_len[k[0]][k[1]][k[2]]
        #     self.seqlen2cnt[sl] = self.seqlen2cnt.get(sl,0) + v

        # self.sorted_bucket_map = dict(sorted(self.runtime_bucket.items(), key=lambda item: item[1], reverse=True))
        
        # print(f'raw_count is {len(self.bucket.raw_count.keys())},runtime_bucket is {len(self.runtime_bucket.keys())}')  
        # print(self.sorted_bucket_map)    
        # print(sum(self.sorted_bucket_map.values()))  
            


    def print_bucket_info(self,bucket_sample_dict: dict) -> int:
         num_total_batch = num_total_samples = 0
         for k,v in bucket_sample_dict.items():
            if not k:
                print(f"  [Warning] Bucket key is empty. Skipping this bucket.")
                continue

            # Check if the sample list 'v' inside the bucket is empty
            if not v:
                print(f"  [Warning] The value 'v' for this bucket {k} is an empty list. This bucket has no samples.")
                size = 0
                num_batch = 0
            else:
                size = len(v)
                print(f"  [+] Number of samples in this bucket (size): {size}")
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            num_total_samples += size
            num_total_batch += num_batch



         return num_total_batch    



        
