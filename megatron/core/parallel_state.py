                                                              

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from typing import Optional

import torch
from collections import defaultdict
from .utils import GlobalMemoryBuffer

                                                                    
_TENSOR_MODEL_PARALLEL_GROUP = None
                                                                    
_PIPELINE_MODEL_PARALLEL_GROUP = None
                                                                                   
_MODEL_PARALLEL_GROUP = None
                  
_EMBEDDING_GROUP = None
                           
_POSITION_EMBEDDING_GROUP = None
                                                       
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
                                                              
                               
_TENSOR_AND_DATA_PARALLEL_GROUP = None
                                                         
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None
         
_TENSOR_CONTEXT_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

                                                            
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_TENSOR_CONTEXT_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None

                                                    
_EMBEDDING_GLOBAL_RANKS = None

                                                             
_POSITION_EMBEDDING_GLOBAL_RANKS = None

                                                                                  
                                                               
_PIPELINE_GLOBAL_RANKS = None

                                                                                       
                                                                          
_DATA_PARALLEL_GLOBAL_RANKS = None

                                                         
_CONTEXT_PARALLEL_GROUP = None
                                                                                   
                                                                        
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

                                                                 
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

                                                        
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

                                                   
_GLOBAL_MEMORY_BUFFER = None

from yunchang.comm.all_to_all import SeqAllToAll4D
             
_MOE_AUX_LOSSES_LOGGING_TRACKER = {}

import torch.distributed as dist
                                    
from megatron.core.logger import logger
_custom_groups_registry = {}
_rank_to_group_name = defaultdict(dict)  
_pre_stage = None
_now_stage = None
_active_group_stack = []
_vae_plan = defaultdict(dict)
_dit_plan = defaultdict(dict)

def reset_pre_stage():
    global _pre_stage
    _pre_stage = None

def get_pre_stage_group():
    return get_context_parallel_group(stage=_pre_stage)

def alltoall_test(stage: str):
    group = get_context_parallel_group(stage=stage)
    bs, seqlen, head_count, hs = 1, 8, 8, 16
    rank = dist.get_rank() % torch.cuda.device_count()
    device = torch.device(f"cuda:{rank}")
    x = torch.full((bs, seqlen, head_count, hs), fill_value=rank, device=device, dtype=torch.bfloat16)
                                          
    SeqAllToAll4D.apply(group, x, 1, 2)
                                         



def get_data_source_ranks(stage: str ) -> list:
    ng_id = get_group_id(stage=stage)
    if stage == "DIT":
        target_dit_group_info = _dit_plan[ng_id]
        if 'data_source_group' in target_dit_group_info:
            source_group_id = target_dit_group_info['data_source_group']
            return _vae_plan[source_group_id]['ranks']

    return []

def switch_scope(stage: str , tensors_to_reshuffle:dict):
    og = get_pre_stage_group()
    ng = get_context_parallel_group(stage=stage)

                          
    og_ranks = dist.get_process_group_ranks(og)
    ng_ranks = dist.get_process_group_ranks(ng)

    o_set = set(og_ranks)
    n_set = set(ng_ranks)

    my_rank = dist.get_rank()
    reshuffled_tensors = {}
                                                                            
    if not n_set.issubset(o_set):
        from my_utils import global_timer

        global_timer.start("broadcast comm overhead")
        data_source_ranks = get_data_source_ranks(stage=stage)


        root_rank = og_ranks[0]
        sender_rank = data_source_ranks[0]
        if my_rank == root_rank and my_rank in data_source_ranks:
                     
            metadata = [{'shape': list(t.shape), 'dtype': t.dtype} for t in tensors_to_reshuffle.values()]
        else:
            metadata = None
        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src = sender_rank, group = ng)
        metadata = metadata_list[0]
        logger.debug(f"metadata = {metadata}")

        if my_rank == root_rank and my_rank in data_source_ranks:
                buffer_list = []
                for t in tensors_to_reshuffle.values():
                    if t.dtype == torch.bfloat16:
                        t = t.view(torch.int16)

                    tensor_bytes = t.cpu().numpy().tobytes()
        
                    byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)
                    buffer_list.append(byte_tensor)
                packed_buffer = torch.cat(buffer_list).to(torch.cuda.current_device())
        else:
                total_bytes = sum(torch.prod(torch.tensor(m['shape'])) * torch.tensor([], dtype=m['dtype']).element_size() for m in metadata)
                packed_buffer = torch.empty(total_bytes, dtype=torch.uint8, device='cuda')
        
        dist.broadcast(packed_buffer, src=sender_rank,group=ng)
                                                                                                

        if my_rank in ng_ranks:
            offset = 0
            tensor_names = list(tensors_to_reshuffle.keys())
            for i, meta in enumerate(metadata):
                num_bytes = torch.prod(torch.tensor(meta['shape'])) * torch.tensor([], dtype=meta['dtype']).element_size()
                tensor_bytes = packed_buffer.narrow(0, int(offset), int(num_bytes)).clone()
                
                recovered_tensor = tensor_bytes.view(meta['dtype']).view(meta['shape'])
                reshuffled_tensors[tensor_names[i]] = recovered_tensor
                
                offset += num_bytes
        global_timer.stop("broadcast comm overhead")        
    else:
        reshuffled_tensors = tensors_to_reshuffle
                                                            
    return reshuffled_tensors


from contextlib import contextmanager

_active_group_stack = []

RESUME_FILE_PATH = "oom_resume_step.txt"
EXIT_CODE_OOM = 1                   

@contextmanager
def use_custom_group(group):
    """
    带 OOM 自动恢复机制的上下文管理器
    """
    if group is None:
        yield
        return
    
    _active_group_stack.append(group)
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "="*60)
        rank_info = os.getenv('RANK', '0')
        print(f"⚠️  [Rank {rank_info}] 捕获到 CUDA Out of Memory 异常！触发自动恢复机制。")
        print(f"   错误信息: {e}")
        
        try:
            from megatron.training import get_args
            from my_utils import set_oom_flag, check_oom_flag
            args = get_args() 
            current_step = getattr(args, 'curr_iteration', 0)
            set_oom_flag()
            is_master = True
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() != 0:
                    is_master = False
            
            if is_master:
                print(f"📝 [Master] 正在将崩溃时的 Iteration {current_step} 写入文件: {RESUME_FILE_PATH}")
                with open(RESUME_FILE_PATH, "w") as f:
                    f.write(str(current_step))
                print("✅ 进度记录完成。")
            else:
                print(f"Waiting for master to save checkpoint info...")
                    
        except Exception as record_err:
            print(f"❌ [Error] 记录恢复点失败: {record_err}")

        try:
            from my_utils import global_timer
            print(f"   执行 global_timer.dump()...")
            global_timer.dump()
        except:
            pass
            
        print(f"🛑 正在以状态码 {EXIT_CODE_OOM} 终止进程，请求 Shell 重启...")
        print("="*60 + "\n")
        
        os._exit(EXIT_CODE_OOM)
        
    finally:
        _active_group_stack.pop()


def push_scope(stage: str, tensors_dict_to_modify:dict = None):
    next_group = get_context_parallel_group(stage=stage)
    global _now_stage
    _now_stage = stage
                                                                                    
    if _pre_stage is not None and _now_stage != _pre_stage:
        new_data_dict = switch_scope(stage=stage, tensors_to_reshuffle=tensors_dict_to_modify)
                                        
        tensors_dict_to_modify.update(new_data_dict)
                       


                                                                                                
    _active_group_stack.append(next_group)

def pop_scope():
    if not _active_group_stack:
        raise RuntimeError("Cannot pop from an empty scope stack.")
    global _pre_stage
    _pre_stage = _now_stage

    _active_group_stack.pop()    
    print(f"after pop _pre_stage = {_pre_stage}")

def register_custom_group(module_name: str, groups_plan: list[dict]):
    """
    创建并注册通信组，并记录每个 rank 对应的组名。
    """

    for item in groups_plan:
        if module_name == 'VAE':
            _vae_plan[item['group_id']] = item
        elif module_name == 'DIT':
            _dit_plan[item['group_id']] = item
        ranks = item['ranks']
        name = item['group_id']
        group = dist.new_group(ranks = ranks)
        if dist.get_rank() in ranks:
            _custom_groups_registry[name] = group
            for r in item['ranks']:
                _rank_to_group_name[r][module_name] = name
            if dist.get_rank() in item:
                print(f"[Rank {dist.get_rank()}] registered group '{name}' pg = {group} with ranks {ranks}")  

    return group


def get_custom_group(name: str):
    """通过组名获取通信组"""
    return _custom_groups_registry.get(name)

def get_group_id(stage: str = None):
    global_rank = dist.get_rank()
    if stage is None:
        raise RuntimeError(f"Rank {global_rank} stage is None")
    return _rank_to_group_name[global_rank][stage]

def get_my_custom_group(stage: str):
    """当前 rank 获取它所属的组"""
    
    global_rank = dist.get_rank()
    name = get_group_id(stage=stage)
    print(f"rank {global_rank}  get_custom_group {name}, world_size = {torch.distributed.get_world_size(_custom_groups_registry[name])}")
    if name is None:
        raise RuntimeError(f"Rank {global_rank} not assigned to any custom {stage} group.")
    return _custom_groups_registry[name]

from t2v_flow.planner import Planner
from typing import List, Dict, Set

_CUSTOM_GROUPS: Dict[frozenset, dist.ProcessGroup] = {}
_CUSTOM_GROUPS_GLOO: Dict[frozenset, dist.ProcessGroup] = {}
_CUSTOM_PYNCCL_COMMS: Dict[frozenset, object] = {}
_CUSTOM_GROUP_STATS: Dict[frozenset, Dict[str, int]] = {}


def _custom_pynccl_enabled() -> bool:
    return os.environ.get("ENABLE_CUSTOM_GROUP_PYNCCL", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def get_custom_pynccl_comm(ranks: List[int]):
    return _CUSTOM_PYNCCL_COMMS.get(frozenset(ranks))


def _ensure_custom_group_stats_entry(group_key: frozenset):
    if group_key not in _CUSTOM_GROUP_STATS:
        _CUSTOM_GROUP_STATS[group_key] = {
            "use_count": 0,
            "last_used_iter": -1,
            "warmup_alltoall_done": 0,
            "warmup_allgather_done": 0,
            "warmup_allreduce_done": 0,
            "warmup_broadcast_done": 0,
        }


def mark_custom_groups_used(group_keys: Set[frozenset], curr_iter: int):
    for group_key in group_keys:
        _ensure_custom_group_stats_entry(group_key)
        _CUSTOM_GROUP_STATS[group_key]["use_count"] += 1
        _CUSTOM_GROUP_STATS[group_key]["last_used_iter"] = curr_iter


def maybe_prune_custom_groups(
    *,
    required_now_groups: Set[frozenset],
    required_future_groups: Set[frozenset],
    curr_iter: int,
    logger,
    gc_interval: int = 3,
    max_cached_groups: int = 32,
    max_local_group_memberships: int = 8,
):
    """
    Periodically prune custom process groups with an LRU+LFU hybrid policy.

    Safety:
    - All ranks must call this function each iteration with logically identical
      `required_now_groups` and `required_future_groups`.
    - On GC iterations we enforce world barriers before/after destroy so no rank
      uses a group while another rank is destroying it.
    """
    if not dist.is_initialized():
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    world_group_key = frozenset(range(world_size))

    required_now = set(required_now_groups)
    required_future = set(required_future_groups)
    required_now.add(world_group_key)
    required_future.add(world_group_key)

    for g in required_now | required_future:
        _ensure_custom_group_stats_entry(g)
    mark_custom_groups_used(required_now, curr_iter)

    gc_interval = max(1, int(gc_interval))
    max_cached_groups = max(1, int(max_cached_groups))
    max_local_group_memberships = max(1, int(max_local_group_memberships))
    should_gc = (curr_iter % gc_interval) == 0
    if not should_gc:
        return

    keep_set = required_now | required_future
    cached_set = set(_CUSTOM_GROUP_STATS.keys()) | set(_CUSTOM_GROUPS.keys())
    cached_set.add(world_group_key)
    cached_non_world = {g for g in cached_set if g != world_group_key}

    local_memberships = sum(1 for g in cached_non_world if rank in g)
    global_excess = max(0, len(cached_non_world) - max_cached_groups)
    local_excess = max(0, local_memberships - max_local_group_memberships)
    sync_device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    local_excess_tensor = torch.tensor([local_excess], dtype=torch.int64, device=sync_device)
    dist.all_reduce(local_excess_tensor, op=dist.ReduceOp.MAX, group=None)
    global_local_excess = int(local_excess_tensor.item())
    eviction_budget = max(global_excess, global_local_excess)

    if eviction_budget <= 0:
        if rank == 0:
            logger.info(
                f"ParallelState GC@iter{curr_iter}: cached={len(cached_non_world)}, "
                f"local_memberships(rank0)={local_memberships}, no eviction needed."
            )
        return

    evictable = [g for g in cached_non_world if g not in keep_set]
    if not evictable:
        if rank == 0:
            logger.info(
                f"ParallelState GC@iter{curr_iter}: eviction budget={eviction_budget}, "
                "but all cached groups are protected by keep-set."
            )
        return

    def _evict_key(g: frozenset):
        stat = _CUSTOM_GROUP_STATS.get(g, None)
        if stat is None:
            return (-1, 0, -len(g), tuple(sorted(g)))
        return (
            stat.get("last_used_iter", -1),
            stat.get("use_count", 0),
            -len(g),
            tuple(sorted(g)),
        )

    evict_list = sorted(evictable, key=_evict_key)[:eviction_budget]
    if not evict_list:
        return

    evict_signature = tuple(tuple(sorted(g)) for g in evict_list)
    gathered_signatures = [None] * world_size
    dist.all_gather_object(gathered_signatures, evict_signature, group=None)
    if any(sig != gathered_signatures[0] for sig in gathered_signatures):
        logger.warning(
            f"ParallelState GC@iter{curr_iter}: inconsistent evict lists across ranks; "
            "skip this GC round for safety."
        )
        return

    dist.barrier()
    for group_key in evict_list:
        group_handle = _CUSTOM_GROUPS.get(group_key, None)
        if group_handle is not None:
            try:
                dist.destroy_process_group(group_handle)
            except Exception as e:
                logger.warning(f"Failed to destroy group {group_key}: {e}")

        gloo_group = _CUSTOM_GROUPS_GLOO.get(group_key, None)
        if gloo_group is not None:
            try:
                dist.destroy_process_group(gloo_group)
            except Exception as e:
                logger.warning(f"Failed to destroy gloo group {group_key}: {e}")

        _CUSTOM_GROUPS.pop(group_key, None)
        _CUSTOM_GROUPS_GLOO.pop(group_key, None)
        _CUSTOM_PYNCCL_COMMS.pop(group_key, None)
        _CUSTOM_GROUP_STATS.pop(group_key, None)
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rank == 0:
        logger.info(
            f"ParallelState GC@iter{curr_iter}: evicted={len(evict_list)}, "
            f"cached_after={len(set(_CUSTOM_GROUP_STATS.keys()) - {world_group_key})}, "
            f"keep_now={len(required_now_groups)}, keep_future={len(required_future_groups)}."
        )


def destroy_custom_groups(logger):
    """
    销毁 _CUSTOM_GROUPS 中存储的所有自定义通信组，并清空字典。
    必须在所有 rank 都不再使用旧组通信后调用。
    """
    if not dist.is_initialized():
        return

    dist.barrier()

    logger.info("ParallelState: Destroying custom groups from previous iteration...")
    
    keys_to_remove = []
    
    for group_key, group_handle in _CUSTOM_GROUPS.items():
        if group_handle is None:
            continue
        
        try:
            dist.destroy_process_group(group_handle)
        except Exception as e:
            logger.warning(f"Failed to destroy group {group_key}: {e}")

        gloo_group = _CUSTOM_GROUPS_GLOO.get(group_key)
        if gloo_group is not None:
            try:
                dist.destroy_process_group(gloo_group)
            except Exception as e:
                logger.warning(
                    f"Failed to destroy gloo group {group_key}: {e}"
                )

        keys_to_remove.append(group_key)

    for k in keys_to_remove:
        del _CUSTOM_GROUPS[k]
        _CUSTOM_GROUPS_GLOO.pop(k, None)
        _CUSTOM_PYNCCL_COMMS.pop(k, None)
        _CUSTOM_GROUP_STATS.pop(k, None)
        
                             

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("ParallelState: Custom groups destroyed.")


def warmup_nccl_group_alltoall(
    nccl_group,
    group_ranks,
    logger,
    num_iters: int = 1,
    device: torch.device = None,
):
    """
    对新建的 NCCL group 做 all-to-all warmup，
    强制触发 NCCL communicator / channel 初始化。
    """
    if nccl_group is None:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank()
    if rank not in group_ranks:
        return

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    group_size = len(group_ranks)

    send = torch.ones(
        group_size, device=device, dtype=torch.bfloat16
    )
    recv = torch.empty_like(send)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

    for i in range(num_iters):
        dist.all_to_all_single(
            output=recv,
            input=send,
            group=nccl_group,
        )

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

def warmup_nccl_group_allgather(
    nccl_group,
    group_ranks,
    logger,
    num_iters: int = 1,
    device: torch.device = None,
):
    """
    对新建的 NCCL group 做 all-gather warmup，
    用于覆盖训练组中可能出现的 all_gather 通路初始化。
    """
    if nccl_group is None:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank()
    if rank not in group_ranks:
        return

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    tensor = torch.ones(1, device=device, dtype=torch.float32)
    gather_list = [torch.empty_like(tensor) for _ in range(len(group_ranks))]

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

    for _ in range(num_iters):
        dist.all_gather(gather_list, tensor, group=nccl_group)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)


def warmup_nccl_group_allreduce(
    nccl_group,
    group_ranks,
    logger,
    num_iters: int = 1,
    device: torch.device = None,
):
    """
    对新建的 NCCL group 做 all-reduce warmup，
    主要用于梯度聚合组，触发 all-reduce 通路初始化。
    """
    if nccl_group is None:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank()
    if rank not in group_ranks:
        return

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    tensor = torch.ones(1, device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=nccl_group)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

def warmup_nccl_group_broadcast(
    nccl_group,
    group_ranks,
    logger,
    num_iters: int = 1,
    device: torch.device = None,
):
    """
    对新建的梯度 NCCL group 做 broadcast warmup，
    触发广播通路初始化，避免第一次真实梯度广播抖动。
    """
    if nccl_group is None:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank()
    if rank not in group_ranks:
        return

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    src_rank = group_ranks[0]
    tensor = torch.ones(1, device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

    for _ in range(num_iters):
        dist.broadcast(tensor, src=src_rank, group=nccl_group)

    torch.cuda.synchronize()
    dist.barrier(group=nccl_group)

def initialize_custom_groups(planner: Planner, logger):
    """
    根据Planner的规划，创建所有自定义的通信组。
    此函数是健壮且幂等的，通过全局共识机制避免重复创建，
    并能智能处理与默认世界组相同的组。
    它采用了“特殊情况优先、本地报告、全局共识、集体创建、本地过滤”的策略。

    Args:
        planner (Planner): 已初始化好的Planner实例。
    """
    if not dist.is_initialized():
        return
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    all_required_groups: Set[frozenset] = planner.get_all_required_gpu_groups()
    grad_required_groups: Set[frozenset] = planner.get_extra_required_groups()
    task_required_groups: Set[frozenset] = set()
    planner_tasks = getattr(planner, "_tasks", None)
    if planner_tasks:
        for task in planner_tasks.values():
            task_required_groups.add(frozenset(task.gpus))

    world_group_frozenset = frozenset(range(world_size))
    if world_group_frozenset in all_required_groups:
        if rank == 0:
            logger.info("ParallelState: Detected request for the default world group. Mapping it directly without creation.")
        _CUSTOM_GROUPS[world_group_frozenset] = None
        _ensure_custom_group_stats_entry(world_group_frozenset)
        all_required_groups.remove(world_group_frozenset)
        grad_required_groups.discard(world_group_frozenset)

    for g in all_required_groups:
        _ensure_custom_group_stats_entry(g)

    my_existing_groups = [
        g for g in all_required_groups if rank in g and g in _CUSTOM_GROUPS
    ]

    all_processes_knowledge = [None] * world_size
    dist.all_gather_object(all_processes_knowledge, my_existing_groups, group=None)

    truly_created_groups = set()
    for group_candidate in all_required_groups:
        is_truly_created = all(
            group_candidate in all_processes_knowledge[member_rank]
            for member_rank in group_candidate
        )
        if is_truly_created:
            truly_created_groups.add(group_candidate)
    
    groups_to_create = all_required_groups.difference(truly_created_groups)

    if not groups_to_create:
        if rank == 0:
            logger.info(
                "ParallelState: All other custom groups already exist or no new groups requested. "
                "Initialization skipped."
            )
    else:
        if rank == 0:
            logger.info(
                f"ParallelState: Collectively creating {len(groups_to_create)} new custom groups..."
            )
        from megatron.training import get_args
        args = get_args()
        timeout = timedelta(minutes=args.distributed_timeout_minutes)
        for group_ranks_set in sorted(
            list(groups_to_create), key=lambda s: tuple(sorted(list(s)))
        ):
            group_ranks_list = sorted(list(group_ranks_set))

            new_group = dist.new_group(ranks=group_ranks_list, timeout=timeout)
            gloo_group = dist.new_group(
                ranks=group_ranks_list, timeout=timeout, backend="gloo"
            )
            if rank in group_ranks_set:
                _CUSTOM_GROUPS[group_ranks_set] = new_group
                _CUSTOM_GROUPS_GLOO[group_ranks_set] = gloo_group
                if _custom_pynccl_enabled():
                    if torch.cuda.is_available():
                        try:
                            from megatron.core.distributed.device_communicators import (
                                PyNcclCommunicator,
                            )

                            device = torch.device(
                                f"cuda:{torch.cuda.current_device()}"
                            )
                            _CUSTOM_PYNCCL_COMMS[group_ranks_set] = PyNcclCommunicator(
                                group=gloo_group,
                                device=device,
                                use_current_stream=False,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Custom PyNcclCommunicator init failed for group {group_ranks_list}: {e}"
                            )
                    else:
                        logger.warning(
                            "ENABLE_CUSTOM_GROUP_PYNCCL is set but CUDA is not available."
                        )

    local_required_groups = [
        g for g in all_required_groups if rank in g and g in _CUSTOM_GROUPS
    ]
    for group_ranks_set in sorted(local_required_groups, key=lambda s: tuple(sorted(list(s)))):
        group_handle = _CUSTOM_GROUPS[group_ranks_set]
        group_ranks_list = sorted(list(group_ranks_set))
        _ensure_custom_group_stats_entry(group_ranks_set)
        stat = _CUSTOM_GROUP_STATS[group_ranks_set]
        if group_ranks_set in grad_required_groups:
            if stat.get("warmup_allreduce_done", 0) == 0:
                warmup_nccl_group_allreduce(
                    nccl_group=group_handle,
                    group_ranks=group_ranks_list,
                    logger=logger,
                    num_iters=1,
                )
                stat["warmup_allreduce_done"] = 1
            if stat.get("warmup_broadcast_done", 0) == 0:
                warmup_nccl_group_broadcast(
                    nccl_group=group_handle,
                    group_ranks=group_ranks_list,
                    logger=logger,
                    num_iters=1,
                )
                stat["warmup_broadcast_done"] = 1
        if group_ranks_set in task_required_groups:
            if stat.get("warmup_alltoall_done", 0) == 0:
                warmup_nccl_group_alltoall(
                    nccl_group=group_handle,
                    group_ranks=group_ranks_list,
                    logger=logger,
                    num_iters=1,
                )
                stat["warmup_alltoall_done"] = 1
            if stat.get("warmup_allgather_done", 0) == 0:
                warmup_nccl_group_allgather(
                    nccl_group=group_handle,
                    group_ranks=group_ranks_list,
                    logger=logger,
                    num_iters=1,
                )
                stat["warmup_allgather_done"] = 1
                   
                                                                        


def get_custom_group(ranks: List[int]) -> dist.ProcessGroup:
    """
    根据给定的ranks列表，获取一个已经创建好的自定义通信组。

    Args:
        ranks (List[int]): 组成一个任务的GPU rank列表。

    Returns:
        dist.ProcessGroup: 对应的NCCL通信组。
    """
    key = frozenset(ranks)
    if key not in _CUSTOM_GROUPS:
        raise RuntimeError(
            f"Rank {dist.get_rank()} is trying to get a custom group for ranks {ranks}, "
            f"but it was not found in its local cache. This indicates a logic error."
        )
    if  _CUSTOM_GROUPS[key] is None:
        print(f"Rank {dist.get_rank()} ======================= get default world group ====================")
        return dist.distributed_c10d._get_default_group()
    return _CUSTOM_GROUPS[key]


def list_custom_groups():
    return list(_custom_groups_registry.keys())


def clear_custom_groups():
    _custom_groups_registry.clear()
    _rank_to_group_name.clear()

def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
                                                         
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    timeout = timedelta(minutes=distributed_timeout_minutes)
                          
                                     
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks_with_cp = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups

        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(
                start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
            )
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            group_with_cp = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
            )
            group_with_cp_gloo = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, backend="gloo"
            )
            if rank in ranks_with_cp:
                _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

                                      
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
                                                                                        
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

                                        
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):

            start_rank = (
                i * num_pipeline_model_parallel_groups
                + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                i * num_pipeline_model_parallel_groups
                + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

                                      
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

                                             
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

                                                                   
                                                                  
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
                                                              
                                 
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

                                              
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
    num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    for i in range(num_tensor_and_data_groups_with_cp):
        start_rank = i * tensor_and_data_group_size_with_cp
        end_rank = start_rank + tensor_and_data_group_size_with_cp
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

        for j in range(context_parallel_size):
            ranks = []
            for k in range(data_parallel_size):
                start_rank = (
                    i * tensor_and_data_group_size_with_cp
                    + j * tensor_model_parallel_size
                    + k * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = start_rank + tensor_model_parallel_size
                ranks = ranks + list(range(start_rank, end_rank))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP = group

                                                  
    global _TENSOR_CONTEXT_PARALLEL_GROUP
    assert (
        _TENSOR_CONTEXT_PARALLEL_GROUP is None
    ), 'Tensor + context parallel group is already initialized'
    tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
    num_tensor_and_context_groups: int = world_size // tensor_and_context_group_size
    print(f"world_size: {world_size}, {tensor_and_context_group_size}")
    for i in range(num_tensor_and_context_groups):
        start_rank = i * tensor_and_context_group_size
        end_rank = start_rank + tensor_and_context_group_size
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_CONTEXT_PARALLEL_GROUP = group

                                               
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
    tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    num_expert_groups: int = data_parallel_size // expert_model_parallel_size
    for i in range(num_tensor_and_data_groups):
        for j in range(num_expert_groups):
                         
            start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
            end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group
            for k in range(tensor_model_parallel_size):
                ranks = range(
                    start_rank + k, end_rank, tensor_model_parallel_size
                )
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _EXPERT_MODEL_PARALLEL_GROUP = group

    for i in range(num_tensor_and_data_groups):
        start_rank = i * tensor_and_data_group_size
        end_rank = (i + 1) * tensor_and_data_group_size
        for j in range(tensor_and_expert_group_size):
            ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

                                     
                                                                              
                                                                                
                             
    _set_global_memory_buffer()


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn(
        "is_unitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_context_parallel_group(check_initialized=True):
    """Get the tensor context parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_CONTEXT_PARALLEL_GROUP is not None
        ), 'tensor context parallel group is not initialized'
    return _TENSOR_CONTEXT_PARALLEL_GROUP

def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True, stage: str = None):
    """Get the context parallel group the caller rank belongs to."""
    try:
                                                                               
        if len(_active_group_stack) > 0:
                                                                                                                                                                     
            return _active_group_stack[-1]
        return get_my_custom_group(stage=stage)
    except Exception as e:
        print(f"[Rank {dist.get_rank()}] Warning: no custom group assigned, fallback to static CONTEXT_PARALLEL_GROUP.  error_message is {e}")
        if check_initialized:
            assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
        return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_expert_model_parallel_group():
    assert (
        _EXPERT_MODEL_PARALLEL_GROUP is not None
    ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is not None
    ), 'data modulo expert parallel group-gloo is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


def set_expert_model_parallel_world_size(world_size):
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_context_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_context_parallel_group())

def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_tensor_context_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    if _MPU_TENSOR_CONTEXT_PARALLEL_RANK is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_context_parallel_group())

def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def get_tensor_context_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_context_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_context_parallel_world_size(stage: str = None):
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group(stage))
    else:
        return 0


def get_context_parallel_rank(stage: str = None):
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group(stage))
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None
    global _TENSOR_CONTEXT_PARALLEL_GROUP
    _TENSOR_CONTEXT_PARALLEL_GROUP = None
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    _MPU_TENSOR_CONTEXT_PARALLEL_RANK= None
