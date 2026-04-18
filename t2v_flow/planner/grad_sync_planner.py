from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, FrozenSet, Optional, Iterable, Tuple
import functools
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GradSyncPlan:
    # 代表 all-reduce 使用的组 key
    # 若能复用已有组，则 rep_group_key ∈ existing_group_keys
    # 否则为 frozenset(reps)（需要新建，且只新建这一组）
    rep_group_key: FrozenSet[int]
    reps: List[int]  # rep_group_key 的成员列表（排序）

    # 广播计划：rep -> (bcast_group_key, bcast_ranks)
    # 注意：bcast_group_key 要么是 existing_group_keys 里的某个组，要么是 None（表示 world group）
    bcast_plan: Dict[int, Tuple[Optional[FrozenSet[int]], List[int]]]

    reuse_rep_group: bool
    created_groups: List[FrozenSet[int]]  # 需要新建的 groups（只可能包含 rep_group_key）


# ---------------------------------------------------------------------------
# 数据提取
# ---------------------------------------------------------------------------

def _extract_rank_to_blocks(
    per_rank_schedules: Dict[int, List],
    world_size: int,
) -> Dict[int, Set[str]]:
    """
    从 per_rank_schedules 提取 rank -> data_block_id 集合（统计所有训练任务）。
    - DIT / DiT：block_id = data_source_task 名称
    - FULL：      block_id = "__full__" + task.name（每个 FULL task 独立 block）

    [Fix-4] task_type 统一转大写比较，避免 "DiT" vs "DIT" 大小写不一致漏统计。
    """
    rank_to_blocks: Dict[int, Set[str]] = {r: set() for r in range(world_size)}
    for r in range(world_size):
        for task in per_rank_schedules.get(r, []):
            args = getattr(task, "args", None) or task.get("args", {})
            task_type = args.get("task_type", "").upper()

            if task_type == "DIT":
                block = args.get("data_source_task")
                if block is not None:
                    rank_to_blocks[r].add(str(block))
            elif task_type == "FULL":
                # FULL task：把 task name 作为 block key，使所有参与 rank 都进入 rep 候选
                task_name = getattr(task, "name", None) or task.get("name", "")
                rank_to_blocks[r].add(f"__full__{task_name}")
    return rank_to_blocks


def _group_to_blocks(
    existing_group_keys: Set[FrozenSet[int]],
    rank_to_blocks: Dict[int, Set[str]],
) -> Dict[FrozenSet[int], Set[str]]:
    out: Dict[FrozenSet[int], Set[str]] = {}
    for g in existing_group_keys:
        covered = set()
        for r in g:
            covered |= rank_to_blocks.get(r, set())
        out[g] = covered
    return out


# ---------------------------------------------------------------------------
# S0：单个已有组覆盖全集
# ---------------------------------------------------------------------------

def _choose_single_group_full_cover(
    g2b: Dict[FrozenSet[int], Set[str]],
    U: Set[str],
) -> Optional[FrozenSet[int]]:
    """
    在已有组中找一个能覆盖全集 U 的组。
    [Fix-3] tie-break 改为选 |g| 最小的（更少通信量），原先是选最大的（错误）。
    """
    best = None
    best_size = None
    for g, cov in g2b.items():
        if cov >= U:
            if best_size is None or len(g) < best_size:
                best = g
                best_size = len(g)
    return best


# ---------------------------------------------------------------------------
# S1：精确最小集合覆盖（取代原来有 Bug 的 _find_combo_cover_reps_from_groups）
# ---------------------------------------------------------------------------

def _min_cover_reps(
    rank_to_blocks: Dict[int, Set[str]],
    U: Set[str],
    world_size: int,
    existing_group_keys: Set[FrozenSet[int]],
) -> Set[int]:
    """
    [Fix-1 & Fix-2] 用精确最小集合覆盖替代有偏差的 greedy。

    算法：
    1. 按 block-set 等价类去重：block-set 相同的 rank 只需一个代表。
    2. 在 unique_blocksets 上做精确最小集合覆盖（迭代加深 + 回溯剪枝）。
    3. 每个获胜 block-set 选一个代表 rank：
       优先选已有执行组成员（提升 Phase-1 all-reduce 复用概率）。
    4. 若极端情况下回溯失败，退回改进版 greedy。
    """
    # Step 1: 去重
    blockset_to_ranks: Dict[FrozenSet[str], List[int]] = defaultdict(list)
    for r in range(world_size):
        bs = frozenset(rank_to_blocks[r])
        if bs:
            blockset_to_ranks[bs].append(r)

    unique_blocksets = list(blockset_to_ranks.keys())
    if not unique_blocksets:
        return set()

    # Step 2: 精确最小集合覆盖（对 unique_blocksets 做回溯）
    universe = sorted(U)
    bit_of = {b: i for i, b in enumerate(universe)}
    full_mask = (1 << len(universe)) - 1

    indexed_blocksets = sorted(
        unique_blocksets,
        key=lambda bs: (-len(bs), tuple(sorted(bs))),
    )
    cover_masks: List[int] = []
    for bs in indexed_blocksets:
        mask = 0
        for b in bs:
            if b in bit_of:
                mask |= 1 << bit_of[b]
        cover_masks.append(mask)

    bit_to_candidates: Dict[int, List[int]] = defaultdict(list)
    for idx, mask in enumerate(cover_masks):
        m = mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            bit_to_candidates[bit_idx].append(idx)
            m ^= lsb

    INF = 10**9

    @functools.lru_cache(maxsize=None)
    def _solve_min_sets(uncovered_mask: int) -> int:
        if uncovered_mask == 0:
            return 0
        pivot_bit = (uncovered_mask & -uncovered_mask).bit_length() - 1
        best = INF
        for idx in bit_to_candidates.get(pivot_bit, []):
            reduced = uncovered_mask & ~cover_masks[idx]
            if reduced == uncovered_mask:
                continue
            cand = 1 + _solve_min_sets(reduced)
            if cand < best:
                best = cand
        return best

    best_cover_indices: List[int] = []
    uncovered_mask = full_mask
    while uncovered_mask:
        pivot_bit = (uncovered_mask & -uncovered_mask).bit_length() - 1
        best_idx = None
        best_reduced = None
        best_score = INF
        for idx in bit_to_candidates.get(pivot_bit, []):
            reduced = uncovered_mask & ~cover_masks[idx]
            if reduced == uncovered_mask:
                continue
            score = 1 + _solve_min_sets(reduced)
            if score < best_score:
                best_idx = idx
                best_reduced = reduced
                best_score = score
        if best_idx is None:
            best_cover_indices = []
            break
        best_cover_indices.append(best_idx)
        uncovered_mask = best_reduced

    if not best_cover_indices:
        # 极端情况退回 greedy
        return _greedy_set_cover_reps(rank_to_blocks, U, world_size)

    winning_blocksets = [indexed_blocksets[i] for i in best_cover_indices]

    # Step 3: 每个 block-set 选 rank，优先属于已有执行组的 rank
    existing_ranks: Set[int] = set()
    for g in existing_group_keys:
        existing_ranks |= set(g)

    reps: Set[int] = set()
    for bs in winning_blocksets:
        candidates = sorted(blockset_to_ranks[bs])
        in_existing = [r for r in candidates if r in existing_ranks]
        pick = min(in_existing) if in_existing else candidates[0]
        reps.add(pick)

    return reps


def _greedy_set_cover_reps(
    rank_to_blocks: Dict[int, Set[str]],
    U: Set[str],
    world_size: int,
) -> Set[int]:
    """改进版 greedy（保留作 fallback）：gain 相同时优先选 block-set 与已选 reps 重叠最少的 rank。"""
    remaining = set(U)
    reps: Set[int] = set()
    covered_blocks: Set[str] = set()
    while remaining:
        best_r = None
        best_gain = -1
        best_overlap = float("inf")
        for r in range(world_size):
            if r in reps:
                continue
            gain = len(rank_to_blocks[r] & remaining)
            if gain > best_gain or (
                gain == best_gain and
                len(rank_to_blocks[r] & covered_blocks) < best_overlap
            ):
                best_gain = gain
                best_overlap = len(rank_to_blocks[r] & covered_blocks)
                best_r = r
        if best_r is None or best_gain <= 0:
            break
        reps.add(best_r)
        covered_blocks |= rank_to_blocks[best_r]
        remaining -= rank_to_blocks[best_r]
    return reps


# ---------------------------------------------------------------------------
# Phase-2 广播计划
# ---------------------------------------------------------------------------

def _assign_rank_to_rep_by_overlap(
    reps: List[int],
    rank_to_blocks: Dict[int, Set[str]],
    world_size: int,
) -> Dict[int, List[int]]:
    """
    把每个 rank 分配给 overlap 最大的 rep，得到 rep -> ranks。
    tie 时选 rank_to_blocks 覆盖规模最大的 rep（广播范围更合理）。
    """
    rep_to_ranks: Dict[int, List[int]] = {rep: [] for rep in reps}
    for r in range(world_size):
        best_rep = max(
            reps,
            key=lambda rep: (
                len(rank_to_blocks[r] & rank_to_blocks[rep]),
                len(rank_to_blocks[rep]),   # tie: 选覆盖更大的 rep
            ),
        )
        rep_to_ranks[best_rep].append(r)
    return rep_to_ranks


def _choose_reusable_bcast_group_or_world(
    rep: int,
    target_ranks: Iterable[int],
    existing_group_keys: Set[FrozenSet[int]],
) -> Tuple[Optional[FrozenSet[int]], List[int]]:
    """
    broadcast 不新建组：优先复用一个已有组 g，使得 (target_ranks ∪ {rep}) ⊆ g。
    若存在多个，选 |g| 最小的（更精确，减少无关 rank 的参与）。
    若不存在，则返回 (None, list(target_ranks))，表示用 world group 广播。
    """
    target = set(target_ranks)
    required = target | {rep}
    best_g = None
    best_size = None

    for g in existing_group_keys:
        if required.issubset(g):
            if best_size is None or len(g) < best_size:
                best_g = g
                best_size = len(g)

    if best_g is not None:
        return best_g, sorted(list(best_g))

    # 兜底：world group（None）
    return None, sorted(list(required))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_group_aware_grad_sync_plan(
    per_rank_schedules: Dict[int, List],
    world_size: int,
    existing_group_keys: Set[FrozenSet[int]],
    max_combo_k: int = 2,  # 保留参数兼容性，不再使用
) -> GradSyncPlan:
    """
    构建两阶段梯度同步计划：
    - Phase-1 all-reduce：rep 组（最小规模，优先复用已有组，至多新建 1 组）
    - Phase-2 broadcast：每个 rep 广播给其负责的 rank 子集（永不新建组）

    修复列表（cc/perf-improvements）：
    [Fix-1] 用精确最小集合覆盖替代原始贪心，保证 rep 数量最优
    [Fix-2] 移除 _find_combo_cover_reps_from_groups（语义错误：把整个 group 所有 rank 当 rep）
    [Fix-3] _choose_single_group_full_cover 改为选最小组（原来选最大）
    [Fix-4] task_type 大小写统一，FULL task 也纳入统计
    [Fix-5] 无训练任务时返回空计划（原来做 world-size 路 all-reduce）
    """
    rank_to_blocks = _extract_rank_to_blocks(per_rank_schedules, world_size)

    # blocks universe
    U: Set[str] = set()
    for s in rank_to_blocks.values():
        U |= s

    # [Fix-5] 无训练任务（U 为空）→ 返回空计划，调用方应跳过 sync
    if not U:
        return GradSyncPlan(
            rep_group_key=frozenset(),
            reps=[],
            bcast_plan={},
            reuse_rep_group=True,
            created_groups=[],
        )

    # existing group coverage
    g2b = _group_to_blocks(existing_group_keys, rank_to_blocks)
    # S1: [Fix-1 & Fix-2] 精确最小集合覆盖选 reps（主目标：rep 数最小）
    candidate_reps = _min_cover_reps(rank_to_blocks, U, world_size, existing_group_keys)
    candidate_rep_group = frozenset(int(x) for x in candidate_reps)
    min_rep_count = len(candidate_rep_group)

    # S0 作为 tie-break：若已有单组全覆盖且组大小 == 最小 reps，优先复用该已有组
    best_full = _choose_single_group_full_cover(g2b, U)
    if best_full is not None and len(best_full) == min_rep_count:
        rep_group_key = best_full
    else:
        rep_group_key = candidate_rep_group
    reps = sorted(list(rep_group_key))

    reuse = rep_group_key in existing_group_keys
    created_groups: List[FrozenSet[int]] = [] if reuse else [rep_group_key]

    rep_to_targets = _assign_rank_to_rep_by_overlap(reps, rank_to_blocks, world_size)
    bcast_plan = {}
    for rep, targets in rep_to_targets.items():
        gkey, ranks = _choose_reusable_bcast_group_or_world(rep, targets, existing_group_keys)
        bcast_plan[rep] = (gkey, ranks)

    return GradSyncPlan(
        rep_group_key=rep_group_key,
        reps=reps,
        bcast_plan=bcast_plan,
        reuse_rep_group=reuse,
        created_groups=created_groups,
    )


# ---------------------------------------------------------------------------
# 执行：实际梯度同步
# ---------------------------------------------------------------------------

def get_all_contiguous_grad_tensors(ddp_model):
    grad_tensors = []
    for buf in ddp_model.buffers:
        grad_tensors.append(buf.grad_data)
    for buf in getattr(ddp_model, "expert_parallel_buffers", []):
        grad_tensors.append(buf.grad_data)
    return grad_tensors


def _get_bucket_grad_tensors(ddp_model) -> List[torch.Tensor]:
    """
    Extract gradient tensors at bucket granularity from the DDP model.

    Megatron's ParamAndGradBuffer splits parameters into multiple Bucket
    objects (each ~40M elements by default). Returning per-bucket tensors
    instead of per-buffer tensors enables finer-grained pipelining between
    Phase-1 AllReduce and Phase-2 Broadcast.

    Falls back to buffer-level tensors when the model has no bucket
    structure (e.g., bucketing is disabled or the model is not Megatron DDP).
    """
    grad_tensors = []
    for buf in ddp_model.buffers:
        buckets = getattr(buf, "buckets", None)
        if buckets and len(buckets) > 1:
            for bucket in buckets:
                grad_tensors.append(bucket.grad_data)
        else:
            # Single bucket or no bucket structure — use full buffer.
            grad_tensors.append(buf.grad_data)
    for buf in getattr(ddp_model, "expert_parallel_buffers", []):
        buckets = getattr(buf, "buckets", None)
        if buckets and len(buckets) > 1:
            for bucket in buckets:
                grad_tensors.append(bucket.grad_data)
        else:
            grad_tensors.append(buf.grad_data)
    return grad_tensors


def _broadcast_bucket(
    grad_tensor: torch.Tensor,
    bcast_plan: Dict[int, Tuple[Optional[FrozenSet[int]], List[int]]],
    rank: int,
    mpu,
) -> List:
    """Issue async broadcast calls for a single grad tensor and return handles."""
    handles = []
    for rep, (bcast_group_key, _bcast_ranks) in bcast_plan.items():
        if bcast_group_key is None:
            h = dist.broadcast(grad_tensor, src=rep, group=None, async_op=True)
            handles.append(h)
        else:
            if rank in bcast_group_key:
                bpg = mpu.get_custom_group(list(bcast_group_key))
                h = dist.broadcast(grad_tensor, src=rep, group=bpg, async_op=True)
                handles.append(h)
    return handles


def sync_grads_with_plan(ddp_model, grad_sync_plan: GradSyncPlan, mpu):
    # [Fix-5] 空计划（无训练任务）直接跳过
    if not grad_sync_plan.reps:
        return

    rank = dist.get_rank()
    is_rep = rank in grad_sync_plan.rep_group_key

    # ---- Bucket-pipelined gradient synchronization ----
    #
    # Instead of running Phase-1 AllReduce on ALL grad tensors, waiting for
    # all to finish, and then running Phase-2 Broadcast on ALL tensors, we
    # pipeline the two phases at bucket granularity:
    #
    #   Bucket 0: AllReduce  ──►  Broadcast
    #   Bucket 1:       AllReduce  ──►  Broadcast
    #   Bucket 2:              AllReduce  ──►  Broadcast
    #                    ...
    #
    # As soon as a bucket's AllReduce completes, its Broadcast is launched
    # immediately — overlapping with the AllReduce of subsequent buckets.
    # This reduces the wall-clock time of the synchronization phase because
    # Phase-2 communication for early buckets runs concurrently with Phase-1
    # communication for later buckets.
    #
    # When the current rank is NOT a representative, Phase-1 is skipped
    # entirely and we only participate in Phase-2 broadcasts (unchanged).

    # All ranks must use the same tensor granularity for broadcast collectives,
    # so everyone extracts at bucket level.  When buckets exist (len > 1),
    # the pipelined path kicks in on rep ranks; non-rep ranks issue all
    # broadcasts eagerly (NCCL queues them on the stream in order).
    grad_tensors = _get_bucket_grad_tensors(ddp_model)
    n_buckets = len(grad_tensors)
    use_pipeline = is_rep and n_buckets > 1

    rep_pg = None
    if is_rep:
        rep_pg = mpu.get_custom_group(grad_sync_plan.rep_group_key)

    # Validate broadcast plan up-front.
    for rep, (bcast_group_key, _) in grad_sync_plan.bcast_plan.items():
        if bcast_group_key is not None and rep not in bcast_group_key:
            raise RuntimeError(
                f"Invalid grad sync plan: rep {rep} not in broadcast group "
                f"{sorted(list(bcast_group_key))}"
            )

    if use_pipeline:
        # -- Pipelined path (rep ranks with multiple buckets) --
        #
        # Overlap AllReduce[i] completion with Broadcast[i] launch while
        # AllReduce[i+1] is already in flight on the NCCL stream.

        trailing_bcast_handles: List = []
        prev_ar_handle = None
        prev_grad_tensor = None

        for g in grad_tensors:
            # Launch async AllReduce for current bucket.
            ar_handle = dist.all_reduce(
                g, op=dist.ReduceOp.SUM, group=rep_pg, async_op=True,
            )

            if prev_ar_handle is not None:
                # Wait for the PREVIOUS bucket's AllReduce to finish ...
                prev_ar_handle.wait()
                # ... and immediately pipeline its Broadcast.
                trailing_bcast_handles.extend(
                    _broadcast_bucket(
                        prev_grad_tensor, grad_sync_plan.bcast_plan, rank, mpu,
                    )
                )

            prev_ar_handle = ar_handle
            prev_grad_tensor = g

        # Drain the last bucket.
        if prev_ar_handle is not None:
            prev_ar_handle.wait()
            trailing_bcast_handles.extend(
                _broadcast_bucket(
                    prev_grad_tensor, grad_sync_plan.bcast_plan, rank, mpu,
                )
            )

        # Wait for all in-flight broadcast operations to complete.
        for h in trailing_bcast_handles:
            h.wait()

    else:
        # -- Non-pipelined path (single bucket, or non-rep rank) --
        # Phase 1: representative all-reduce (only rep ranks participate).
        if is_rep:
            handles = []
            for g in grad_tensors:
                h = dist.all_reduce(
                    g, op=dist.ReduceOp.SUM, group=rep_pg, async_op=True,
                )
                handles.append(h)
            for h in handles:
                h.wait()

        # Phase 2: broadcast from reps to all ranks.
        # Async ops are queued on the NCCL stream in tensor order, matching
        # the issue order on the rep side (pipelined or sequential).
        bcast_handles = []
        for g in grad_tensors:
            bcast_handles.extend(
                _broadcast_bucket(g, grad_sync_plan.bcast_plan, rank, mpu)
            )
        for h in bcast_handles:
            h.wait()
