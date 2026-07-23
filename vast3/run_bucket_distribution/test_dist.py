import os
import time
import torch
import torch.distributed as dist


def init_dist():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def measure_new_group_time(rank, group_ranks, tag):
    if rank in group_ranks:
        torch.cuda.synchronize()
        t0 = time.time()
        group = dist.new_group(ranks=group_ranks)
        # dist.barrier(group=group)
        torch.cuda.synchronize()
        
        elapsed = time.time() - t0
        print(f"[Rank {rank}] Group {tag} created in {elapsed:.6f}s")
        dist.destroy_process_group(group)


def main():
    rank, local_rank, world_size = init_dist()

    # 定义测试的 group
    groups = {
        # "01": [0, 1],
        # "23": [2, 3],
        # "45": [4, 5],
        # "67": [6, 7],
        # "0123": [0, 1, 2, 3],
        # "4567": [4, 5, 6, 7],
        "01234567": list(range(8))
    }

    for tag, group_ranks in groups.items():
        measure_new_group_time(rank, group_ranks, tag)

    # dist.barrier()
    if rank == 0:
        print("✅ Group creation time test complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
