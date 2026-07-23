import math
import torch.distributed as dist


def map_target_fps(
    fps: float,
    max_fps: float,
) -> tuple[float, int]:
    """
    Map fps to a new fps that is less than max_fps.

    Args:
        fps (float): Original fps.
        max_fps (float): Maximum fps.

    Returns:
        tuple[float, int]: New fps and sampling interval.
    """
    if math.isnan(fps):
        return 0, 1
    if fps < max_fps:
        return fps, 1
    sampling_interval = math.ceil(fps / max_fps)
    new_fps = math.floor(fps / sampling_interval)
    return new_fps, sampling_interval


def sync_object_across_devices(obj: any, rank: int=0):
    obj_list = [obj]
    dist.broadcast_object_list(object_list=obj_list, src=rank, device="cuda")
    obj = obj_list[0]
    return obj


def gather_micro_batches_across_gpus(micro_batch, group=None):
    if group is None:
        group = dist.group.WORLD
    print(f'[Rank {dist.get_rank()}] group_size = {dist.get_world_size(group)}')
    output_list = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(output_list, micro_batch, group=group)
    return output_list