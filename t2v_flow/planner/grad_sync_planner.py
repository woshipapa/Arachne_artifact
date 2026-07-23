from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, FrozenSet, Optional, Iterable, Tuple
import functools
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GradSyncPlan:
    # all-reduce group key: an existing execution group when reusable,
    # otherwise frozenset(reps) (the only group ever created)
    rep_group_key: FrozenSet[int]
    reps: List[int]  # sorted members of rep_group_key

    # rep -> (bcast_group_key, ranks); key is an existing group or None (world)
    bcast_plan: Dict[int, Tuple[Optional[FrozenSet[int]], List[int]]]

    reuse_rep_group: bool
    created_groups: List[FrozenSet[int]]  # groups to create (at most the rep group)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _extract_rank_to_blocks(
    per_rank_schedules: Dict[int, List],
    world_size: int,
) -> Dict[int, Set[str]]:
    """Extract rank -> {data_block_id} over all training tasks:
    DIT blocks are keyed by data_source_task, FULL tasks by their own name.
    task_type is compared case-insensitively."""
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
# ---------------------------------------------------------------------------

def _choose_single_group_full_cover(
    g2b: Dict[FrozenSet[int], Set[str]],
    U: Set[str],
) -> Optional[FrozenSet[int]]:
    """Find an existing group covering the full rank set U; ties prefer the
    smallest group (less communication)."""
    best = None
    best_size = None
    for g, cov in g2b.items():
        if cov >= U:
            if best_size is None or len(g) < best_size:
                best = g
                best_size = len(g)
    return best


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _min_cover_reps(
    rank_to_blocks: Dict[int, Set[str]],
    U: Set[str],
    world_size: int,
    existing_group_keys: Set[FrozenSet[int]],
) -> Set[int]:
    """Exact minimum set cover for representative selection:
    1) dedup ranks by block-set equivalence; 2) iterative-deepening
    backtracking over unique block-sets; 3) pick one representative per
    winning block-set, preferring members of existing execution groups
    (better Phase-1 all-reduce reuse); 4) fall back to greedy if the
    search is cut off."""
    blockset_to_ranks: Dict[FrozenSet[str], List[int]] = defaultdict(list)
    for r in range(world_size):
        bs = frozenset(rank_to_blocks[r])
        if bs:
            blockset_to_ranks[bs].append(r)

    unique_blocksets = list(blockset_to_ranks.keys())
    if not unique_blocksets:
        return set()

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
        return _greedy_set_cover_reps(rank_to_blocks, U, world_size)

    winning_blocksets = [indexed_blocksets[i] for i in best_cover_indices]

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
    """Greedy fallback: on equal gain prefer the rank whose block-set
    overlaps least with already-chosen representatives."""
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
# ---------------------------------------------------------------------------

def _assign_rank_to_rep_by_overlap(
    reps: List[int],
    rank_to_blocks: Dict[int, Set[str]],
    world_size: int,
) -> Dict[int, List[int]]:
    """Assign every rank to the representative with the largest block
    overlap; ties prefer the rep with the larger block coverage."""
    rep_to_ranks: Dict[int, List[int]] = {rep: [] for rep in reps}
    for r in range(world_size):
        best_rep = max(
            reps,
            key=lambda rep: (
                len(rank_to_blocks[r] & rank_to_blocks[rep]),
                len(rank_to_blocks[rep]),
            ),
        )
        rep_to_ranks[best_rep].append(r)
    return rep_to_ranks


def _choose_reusable_bcast_group_or_world(
    rep: int,
    target_ranks: Iterable[int],
    existing_group_keys: Set[FrozenSet[int]],
) -> Tuple[Optional[FrozenSet[int]], List[int]]:
    """Broadcast never creates groups: reuse the smallest existing group
    containing (target_ranks | {rep}); otherwise fall back to the world
    group (returned as None)."""
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

    return None, sorted(list(required))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_group_aware_grad_sync_plan(
    per_rank_schedules: Dict[int, List],
    world_size: int,
    existing_group_keys: Set[FrozenSet[int]],
    max_combo_k: int = 2,  # kept for signature compatibility (unused)
) -> GradSyncPlan:
    """Build the two-phase heterogeneous gradient-sync plan:
    - Phase 1 all-reduce over a minimum representative set (exact set
      cover; reuse an existing execution group when possible, create at
      most one new group);
    - Phase 2 broadcast from each representative to its assigned ranks
      (never creates groups).
    Returns an empty plan when there are no training tasks."""
    rank_to_blocks = _extract_rank_to_blocks(per_rank_schedules, world_size)

    # blocks universe
    U: Set[str] = set()
    for s in rank_to_blocks.values():
        U |= s

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
    candidate_reps = _min_cover_reps(rank_to_blocks, U, world_size, existing_group_keys)
    candidate_rep_group = frozenset(int(x) for x in candidate_reps)
    min_rep_count = len(candidate_rep_group)

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
    # rep -> (bcast_group_key, ranks); key is an existing group or None (world)
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
