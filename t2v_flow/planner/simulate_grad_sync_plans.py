"""

    python t2v_flow/planner/simulate_grad_sync_plans.py [--dir <schedule_dir>] [--verbose]

"""

import sys
import os
import yaml
import glob
import argparse
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, FrozenSet, Optional, Tuple, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import types as _types
if "torch" not in sys.modules:
    _torch_stub = _types.ModuleType("torch")
    _dist_stub = _types.ModuleType("torch.distributed")
    _dist_stub.get_rank = lambda: 0
    _torch_stub.distributed = _dist_stub
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.distributed"] = _dist_stub

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "grad_sync_planner",
    os.path.join(ROOT, "t2v_flow/planner/grad_sync_planner.py"),
)
_gsp = _ilu.module_from_spec(_spec)
sys.modules["grad_sync_planner"] = _gsp
_spec.loader.exec_module(_gsp)

build_group_aware_grad_sync_plan = _gsp.build_group_aware_grad_sync_plan
_extract_rank_to_blocks          = _gsp._extract_rank_to_blocks
_min_cover_reps                  = _gsp._min_cover_reps
GradSyncPlan                     = _gsp.GradSyncPlan


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class _FakeTask:
    def __init__(self, name, gpus, args):
        self.name = name
        self.gpus = gpus
        self.args = args

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_schedule(yaml_path: str):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    tasks = data.get("tasks", []) or []
    all_gpus: Set[int] = set()
    for t in tasks:
        for g in t.get("gpus", []):
            all_gpus.add(int(g))

    if not all_gpus:
        return {}, set(), 0

    world_size = max(all_gpus) + 1
    per_rank_schedules: Dict[int, List] = {r: [] for r in range(world_size)}
    existing_group_keys: Set[FrozenSet[int]] = set()

    for t in tasks:
        gpus = [int(g) for g in t.get("gpus", [])]
        args = t.get("args", {}) or {}
        task_obj = _FakeTask(name=t.get("name", ""), gpus=gpus, args=args)
        existing_group_keys.add(frozenset(gpus))
        for r in gpus:
            per_rank_schedules[r].append(task_obj)

    return per_rank_schedules, existing_group_keys, world_size


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _old_extract_rank_to_blocks(per_rank_schedules, world_size):
    rank_to_blocks = {r: set() for r in range(world_size)}
    for r in range(world_size):
        for task in per_rank_schedules.get(r, []):
            args = getattr(task, "args", None) or task.get("args", {})
            if args.get("task_type") != "DIT":
                continue
            block = args.get("data_source_task")
            if block is not None:
                rank_to_blocks[r].add(str(block))
    return rank_to_blocks


def _old_greedy_set_cover_reps(rank_to_blocks, U, world_size):
    remaining = set(U)
    reps = set()
    while remaining:
        best_r, best_gain = None, -1
        for r in range(world_size):
            gain = len(rank_to_blocks[r] & remaining)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_r is None or best_gain <= 0:
            break
        reps.add(best_r)
        remaining -= rank_to_blocks[best_r]
    return reps


def _old_find_combo(existing_group_keys, g2b, U, max_k=2):
    groups = list(existing_group_keys)
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(groups, k):
            cov, reps_set = set(), set()
            for g in combo:
                cov |= g2b[g]
                reps_set |= set(g)
            if cov == U:
                return reps_set
    return None


def _old_choose_single(g2b, U):
    best, best_size = None, -1
    for g, cov in g2b.items():
        if cov == U and len(g) > best_size:
            best, best_size = g, len(g)
    return best


def _old_assign_to_rep(reps, rank_to_blocks, world_size):
    rep_to_ranks = {rep: [] for rep in reps}
    for r in range(world_size):
        best_rep = max(reps, key=lambda rep: len(rank_to_blocks[r] & rank_to_blocks[rep]))
        rep_to_ranks[best_rep].append(r)
    return rep_to_ranks


def _old_choose_bcast_group(rep, target_ranks, existing_group_keys):
    target = set(target_ranks)
    required = target | {rep}
    best_g, best_size = None, None
    for g in existing_group_keys:
        if required.issubset(g):
            if best_size is None or len(g) < best_size:
                best_g, best_size = g, len(g)
    if best_g is not None:
        return best_g, sorted(list(best_g))
    return None, sorted(list(required))


def build_old_plan(per_rank_schedules, world_size, existing_group_keys):
    rank_to_blocks = _old_extract_rank_to_blocks(per_rank_schedules, world_size)
    U = set()
    for s in rank_to_blocks.values():
        U |= s

    if not U:
        reps = list(range(world_size))
        rep_group_key = frozenset(reps)
        reuse = rep_group_key in existing_group_keys
        return {
            "rep_group_key": rep_group_key,
            "reps": reps,
            "reuse": reuse,
            "n_new_groups": 0 if reuse else 1,
            "bcast_plan": {reps[0]: (None, reps)},
        }

    g2b = {}
    for g in existing_group_keys:
        covered = set()
        for r in g:
            covered |= rank_to_blocks.get(r, set())
        g2b[g] = covered

    best_full = _old_choose_single(g2b, U)
    if best_full is not None:
        rep_group_key = best_full
        reps = sorted(list(rep_group_key))
        rep_to_targets = _old_assign_to_rep(reps, rank_to_blocks, world_size)
        bcast_plan = {}
        for rep, targets in rep_to_targets.items():
            gkey, ranks = _old_choose_bcast_group(rep, targets, existing_group_keys)
            bcast_plan[rep] = (gkey, ranks)
        return {
            "rep_group_key": rep_group_key, "reps": reps,
            "reuse": True, "n_new_groups": 0, "bcast_plan": bcast_plan,
        }

    candidate_reps = _old_find_combo(existing_group_keys, g2b, U)
    if candidate_reps is None:
        candidate_reps = _old_greedy_set_cover_reps(rank_to_blocks, U, world_size)

    rep_group_key = frozenset(int(x) for x in candidate_reps)
    reps = sorted(list(rep_group_key))
    reuse = rep_group_key in existing_group_keys
    rep_to_targets = _old_assign_to_rep(reps, rank_to_blocks, world_size)
    bcast_plan = {}
    for rep, targets in rep_to_targets.items():
        gkey, ranks = _old_choose_bcast_group(rep, targets, existing_group_keys)
        bcast_plan[rep] = (gkey, ranks)
    return {
        "rep_group_key": rep_group_key, "reps": reps,
        "reuse": reuse, "n_new_groups": 0 if reuse else 1,
        "bcast_plan": bcast_plan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(plan_dict_or_obj, world_size: int, existing_group_keys: Set[FrozenSet[int]]) -> Dict:
    if isinstance(plan_dict_or_obj, GradSyncPlan):
        reps = plan_dict_or_obj.reps
        rep_group_key = plan_dict_or_obj.rep_group_key
        bcast_plan = plan_dict_or_obj.bcast_plan
        n_new_groups = len(plan_dict_or_obj.created_groups)
        reuse = plan_dict_or_obj.reuse_rep_group
    else:
        d = plan_dict_or_obj
        reps = d["reps"]
        rep_group_key = d["rep_group_key"]
        bcast_plan = d["bcast_plan"]
        n_new_groups = d["n_new_groups"]
        reuse = d["reuse"]

    n_reps = len(reps)
    rep_group_size = len(rep_group_key)

    rank_bcast_count = defaultdict(int)
    for rep, (bcast_group_key, bcast_ranks) in bcast_plan.items():
        targets = set(bcast_ranks) if bcast_ranks else set()
        if bcast_group_key is not None:
            targets = set(bcast_group_key)
        for r in targets:
            rank_bcast_count[r] += 1

    total_extra_bcast = sum(max(0, c - 1) for c in rank_bcast_count.values())
    max_bcast_per_rank = max(rank_bcast_count.values()) if rank_bcast_count else 0

    n_bcast = len(bcast_plan)
    n_bcast_reuse = sum(1 for gk, _ in bcast_plan.values() if gk is not None)
    n_bcast_world = n_bcast - n_bcast_reuse

    return {
        "n_reps": n_reps,
        "rep_group_size": rep_group_size,
        "reuse_rep_group": reuse,
        "n_new_groups": n_new_groups,
        "total_extra_bcast_participations": total_extra_bcast,
        "max_bcast_per_rank": max_bcast_per_rank,
        "n_bcast_reuse_existing": n_bcast_reuse,
        "n_bcast_world": n_bcast_world,
        "reps": sorted(reps),
    }


def _is_optimal_reps(new_n_reps: int, per_rank_schedules, world_size: int, existing_group_keys) -> Tuple[bool, int]:
    rank_to_blocks = _extract_rank_to_blocks(per_rank_schedules, world_size)
    U = set()
    for s in rank_to_blocks.values():
        U |= s
    if not U:
        return True, 0

    blockset_to_ranks = defaultdict(list)
    for r in range(world_size):
        bs = frozenset(rank_to_blocks[r])
        if bs:
            blockset_to_ranks[bs].append(r)

    unique_blocksets = list(blockset_to_ranks.keys())
    n = len(unique_blocksets)

    if n > 12:
        return True, -1

    optimal = n
    for size in range(1, n + 1):
        found = False
        for combo in itertools.combinations(range(n), size):
            covered = set()
            for i in combo:
                covered |= unique_blocksets[i]
            if covered >= U:
                optimal = size
                found = True
                break
        if found:
            break

    return new_n_reps == optimal, optimal


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run(schedule_dirs: List[str], verbose: bool = False):
    yaml_files = []
    for d in schedule_dirs:
        yaml_files += sorted(glob.glob(os.path.join(d, "*.yaml")))
        yaml_files += sorted(glob.glob(os.path.join(d, "*.yml")))

    if not yaml_files:
        print(f"[WARN] no YAML file found, dirs={schedule_dirs}")
        return

    total = 0
    old_wins = 0
    new_wins = 0
    tie = 0
    new_is_optimal = 0
    old_is_optimal = 0
    total_old_extra_bcast = 0
    total_new_extra_bcast = 0

    COL = "{:<45} {:>5} {:>5} {:>6} {:>5} {:>6} {:>5} {:>8} {:>8} {:>4}"
    header = COL.format(
        "schedule", "ws",
        "old_r", "new_r", "opt",
        "old_ex", "new_ex",
        "old_ng", "new_ng", "ΔBCT"
    )
    SEP = "-" * len(header)
    print(SEP)
    print(header)
    print(SEP)

    for yaml_path in yaml_files:
        per_rank_schedules, existing_group_keys, world_size = load_schedule(yaml_path)
        if world_size == 0:
            continue

        try:
            old_plan = build_old_plan(per_rank_schedules, world_size, existing_group_keys)
            new_plan = build_group_aware_grad_sync_plan(
                per_rank_schedules, world_size, existing_group_keys
            )
        except Exception as e:
            print(f"  [ERR] {yaml_path}: {e}")
            continue

        old_m = _compute_metrics(old_plan, world_size, existing_group_keys)
        new_m = _compute_metrics(new_plan, world_size, existing_group_keys)
        new_opt, optimal_n = _is_optimal_reps(new_m["n_reps"], per_rank_schedules, world_size, existing_group_keys)
        old_opt, _ = _is_optimal_reps(old_m["n_reps"], per_rank_schedules, world_size, existing_group_keys)

        total += 1
        total_old_extra_bcast += old_m["total_extra_bcast_participations"]
        total_new_extra_bcast += new_m["total_extra_bcast_participations"]
        if new_opt and optimal_n != -1:
            new_is_optimal += 1
        if old_opt and optimal_n != -1:
            old_is_optimal += 1

        old_r = old_m["n_reps"]
        new_r = new_m["n_reps"]
        if new_r < old_r:
            new_wins += 1
        elif old_r < new_r:
            old_wins += 1
        else:
            tie += 1

        delta_bct = old_m["total_extra_bcast_participations"] - new_m["total_extra_bcast_participations"]

        fname = os.path.relpath(yaml_path, ROOT)[-44:]
        opt_str = "?" if optimal_n == -1 else f"{optimal_n}{'✓' if new_opt else '✗'}"
        print(COL.format(
            fname, world_size,
            old_r, new_r, opt_str,
            old_m["total_extra_bcast_participations"],
            new_m["total_extra_bcast_participations"],
            old_m["n_new_groups"],
            new_m["n_new_groups"],
            f"+{delta_bct}" if delta_bct >= 0 else str(delta_bct),
        ))

        if verbose and (new_r != old_r or not new_opt or delta_bct != 0):
            print(f"    [OLD] reps={old_m['reps']}  rep_grp_sz={old_m['rep_group_size']}  "
                  f"reuse={old_m['reuse_rep_group']}")
            print(f"    [NEW] reps={new_m['reps']}  rep_grp_sz={new_m['rep_group_size']}  "
                  f"reuse={new_m['reuse_rep_group']}")
            # Show new bcast plan
            for rep, (gk, ranks) in sorted(new_plan.bcast_plan.items()):
                gk_str = f"grp{sorted(list(gk))}" if gk else "world"
                print(f"         rep {rep} -> {gk_str}  ranks={sorted(ranks)}")

    print(SEP)
    print(f"\nStatistics ({total} schedules):")
    print(f"  new plan uses fewer reps (improvement): {new_wins} / {total}  ({100*new_wins/max(1,total):.1f}%)")
    print(f"  identical:                              {tie} / {total}  ({100*tie/max(1,total):.1f}%)")
    print(f"  old plan uses fewer (should not happen):{old_wins} / {total}")
    print(f"  new plan reaches the theoretical optimum:{new_is_optimal} / {total}  ({100*new_is_optimal/max(1,total):.1f}%)")
    print(f"  old plan reaches the theoretical optimum:{old_is_optimal} / {total}  ({100*old_is_optimal/max(1,total):.1f}%)")
    print(f"  broadcast overlap removed by the new plan:{total_old_extra_bcast - total_new_extra_bcast}  "
          f"(old={total_old_extra_bcast}, new={total_new_extra_bcast})")


def main():
    parser = argparse.ArgumentParser(description="Simulate and compare the old vs new grad-sync plan")
    parser.add_argument(
        "--dir", "-d",
        nargs="+",
        default=[
            "t2v_flow/planner/generated_schedules/hunyuan-4nodes/720p/genetic",
            "t2v_flow/planner/generated_schedules/cogvideox/720p/129/flex_sp",
            "t2v_flow/planner/generated_schedules/hunyuan/720p/genetic",
            "t2v_flow/planner/generated_schedules/wan/720p/genetic",
        ],
        help="schedule YAML directory (repeatable)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print the full plan")
    args = parser.parse_args()

    dirs = [os.path.join(ROOT, d) if not os.path.isabs(d) else d for d in args.dir]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        dirs = []
        for root, subdirs, files in os.walk(os.path.join(ROOT, "t2v_flow/planner/generated_schedules")):
            if any(f.endswith(".yaml") or f.endswith(".yml") for f in files):
                dirs.append(root)
        dirs = sorted(set(dirs))[:20]

    print(f"Scanning {len(dirs)} directory/directories:")
    for d in dirs:
        print(f"  {d}")
    print()
    run(dirs, verbose=args.verbose)


if __name__ == "__main__":
    main()
