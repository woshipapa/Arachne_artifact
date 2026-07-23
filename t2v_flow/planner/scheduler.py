import heapq
import pandas as pd
from typing import List, Tuple, Dict, IO
import time
import matplotlib.pyplot as plt
import pandas as pd
import random
import yaml
import os

import functools
from tqdm import tqdm
from collections import namedtuple
import itertools
from .TopologyModel import TopologyModel
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import cProfile

SP_STRATEGIES = [1, 2, 4, 8]

DATASET_EXEC_TIMES_20 = {
    0: {
        "name": "D_97",
        "vae": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "dit": {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94},
    },
    1: {
        "name": "D_121",
        "vae": {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03},
        "dit": {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83},
    },
    2: {
        "name": "D_221",
        "vae": {2: 52.9, 4: 26.64, 8: 16.5},
        "dit": {2: 74.8, 4: 36.5, 8: 18.36},
    },
    3: {
        "name": "D_241",
        "vae": {2: 52.26, 4: 30.01, 8: 17.29},
        "dit": {2: 88.61, 4: 42.7, 8: 21.58},
    },
}


DATASET_EXEC_TIMES = {
    0: {
        "name": "D_97",
        "VAE": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "DiT": {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94},
    },
    1: {
        "name": "D_121",
        "VAE": {1: 52.76, 2: 28.04, 4: 15.7, 8: 10.03},
        "DiT": {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83},
    },
    2: {
        "name": "D_221",
        "VAE": {2: 52.9, 4: 31.60, 8: 16.5},
        "DiT": {2: 74.8, 4: 40.35, 8: 18.36},
    },
    3: {
        "name": "D_241",
        "VAE": {2: 63.77, 4: 30.01, 8: 17.29},
        "DiT": {2: 92.94, 4: 42.7, 8: 21.58},
    },
}
DATASET_EXEC_TIMES_120 = {
    0: {
        "name": "D_000",
        "vae": {1: 37.36, 2: 20.02, 4: 10.73, 8: 5.75},
        "dit": {1: 69.59, 2: 37.29, 4: 19.99, 8: 10.71},
    },
    1: {
        "name": "D_001",
        "vae": {1: 50.31, 2: 26.96, 4: 14.45, 8: 7.74},
        "dit": {1: 78.99, 2: 42.33, 4: 22.68, 8: 12.16},
    },
    2: {
        "name": "D_002",
        "vae": {1: 34.43, 2: 18.45, 4: 9.89, 8: 5.3},
        "dit": {1: 91.41, 2: 48.99, 4: 26.25, 8: 14.07},
    },
    3: {
        "name": "D_003",
        "vae": {1: 46.96, 2: 25.17, 4: 13.49, 8: 7.23},
        "dit": {1: 94.89, 2: 50.85, 4: 27.25, 8: 14.6},
    },
}


def generate_fake_dataset_exec_times(
    num_datasets: int = 4,
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Generate synthetic per-SP execution times (larger SP -> shorter time)."""
    SP_STRATEGIES = [1, 2, 4, 8]
    fake_exec_times = {}

    for i in range(num_datasets):
        name = f"D_{i:03d}"

        base_vae = random.uniform(20, 60)
        base_dit = random.uniform(40, 100)

        vae = {}
        dit = {}
        for idx, sp in enumerate(SP_STRATEGIES):
            vae[sp] = round(base_vae / (sp**0.9), 2)
            dit[sp] = round(base_dit / (sp**0.9), 2)

        fake_exec_times[i] = {
            "name": name,
            "vae": vae,
            "dit": dit,
        }

    return fake_exec_times


def simulate_tasks_variable(num_tasks: int) -> List[Dict[str, List[Tuple[int, float]]]]:
    tasks = []
    for i in range(num_tasks):
        dataset_id = i % 4
        fixed_times = DATASET_EXEC_TIMES[dataset_id]
        name = fixed_times["name"]
        dit_opts = list(fixed_times["DiT"].items())
        vae_opts = list(fixed_times["VAE"].items())
        tasks.append(
            {
                "dit": dit_opts,
                "vae": vae_opts,
                "dataset_id": name,
            }
        )
    return tasks


def simulate_tasks_fixed(num_tasks: int) -> List[Dict[str, List[Tuple[int, float]]]]:
    fixed_times = {
        "vae": {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        "dit": {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17},
    }
    tasks = []
    for _ in range(num_tasks):
        dit_opts = [(k, fixed_times["dit"][k]) for k in SP_STRATEGIES]
        vae_opts = [(k, fixed_times["vae"][k]) for k in SP_STRATEGIES]
        tasks.append({"dit": dit_opts, "vae": vae_opts})
    return tasks


class State:
    def __init__(self, finished, gpu_timeline, total_time, trace, stage_end_time):
        self.finished = finished
        self.gpu_timeline = gpu_timeline
        self.total_time = total_time
        self.trace = trace
        self.stage_end_time = stage_end_time

    def __lt__(self, other):
        return self.total_time < other.total_time


def _generate_split_plans(k: int, machine_groups: Dict[int, List[int]]) -> List[Dict[int, int]]:
    """Enumerate all cross-machine split plans for a k-GPU group."""
    machines = sorted(machine_groups.keys())
    available_counts = {m: len(machine_groups[m]) for m in machines}
    plans = []

    def find_splits(machine_idx, current_k, current_plan):
        if machine_idx == len(machines):
            if current_k == 0: plans.append(current_plan.copy())
            return
        machine_id = machines[machine_idx]
        max_from_this_machine = min(current_k, available_counts[machine_id])
        for i in range(max_from_this_machine + 1):
            remaining_k = current_k - i
            remaining_capacity = sum(available_counts[machines[j]] for j in range(machine_idx + 1, len(machines)))
            if remaining_k > remaining_capacity: continue
            current_plan[machine_id] = i
            find_splits(machine_idx + 1, remaining_k, current_plan)
            del current_plan[machine_id]
    find_splits(0, k, {})
    return [p for p in plans if len(p.values()) > 1 and all(v > 0 for v in p.values())]

def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    sorted_gpus = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
    selected = sorted_gpus[:k]
    start = max(gpu_timeline[i] for i in selected)
    return selected, start


import heapq
from collections import defaultdict, deque

def schedule_makespan_heap_ready(
    all_tasks,          # list of task_key, e.g. [(tid,'VAE'),(tid,'DIT'),...]
    k_map,              # task_key -> sp(k)
    dur_map,            # task_key -> duration
    deps,               # task_key -> list[task_key]
    total_gpus: int,
    pick_rule="kxdur",  # or "lpt"
):
    # ---------- build indegree ----------
    indeg = {t: 0 for t in all_tasks}
    succ = defaultdict(list)
    for t in all_tasks:
        for d in deps.get(t, []):
            succ[d].append(t)
            indeg[t] += 1

    # ---------- ready set ----------
    ready = [t for t in all_tasks if indeg[t] == 0]

    def score(t):
        k = k_map[t]
        d = dur_map[t]
        if pick_rule == "lpt":
            return (d, k)              # long duration first
        else:
            return (k * d, d, k)       # resource-time first

    # ---------- GPU slots heap ----------
    slots = [(0.0, i) for i in range(total_gpus)]
    heapq.heapify(slots)

    finished = {}
    makespan = 0.0

    while ready:
        # pick best task among ready (O(|ready|), but |ready|<=2*T, T small in your use)
        t = max(ready, key=score)
        ready.remove(t)

        k = k_map[t]
        if k > total_gpus:
            return float("inf")

        dep_finish = 0.0
        for d in deps.get(t, []):
            dep_finish = max(dep_finish, finished[d])

        chosen = [heapq.heappop(slots) for _ in range(k)]
        start_slots = max(tt for tt, _ in chosen)
        start = max(start_slots, dep_finish)

        end = start + dur_map[t]
        finished[t] = end
        makespan = max(makespan, end)

        for _, sid in chosen:
            heapq.heappush(slots, (end, sid))

        # release successors
        for nx in succ[t]:
            indeg[nx] -= 1
            if indeg[nx] == 0:
                ready.append(nx)

    # if not all scheduled => cycle
    if len(finished) != len(all_tasks):
        return float("inf")

    return makespan


def pick_next_ready(ready, k_map, dur_map):
    return max(ready, key=lambda t: (k_map[t] * dur_map[t], dur_map[t], k_map[t]))

import random

def local_improve(chrom, sp_options_per_gene, eval_fn, tries=3):
    best = chrom
    best_val = eval_fn(chrom)
    L = len(chrom)

    for _ in range(tries):
        cand = best[:]
        j = random.randrange(L)
        cand[j] = random.choice(sp_options_per_gene[j])
        val = eval_fn(cand)
        if val < best_val:
            best, best_val = cand, val
    return best, best_val


import heapq

def find_gpus_with_cascade_awareness(
    gpu_timeline: List[float], 
    k: int, 
    preferred_gpus: List[int] = None
) -> Tuple[List[int], float]:
    """GPU selector with inter-cascade affinity: prefer `preferred_gpus`
    (the producer VAE's placement) and fill the remainder with the
    earliest-free GPUs; returns (gpus, ready_time)."""
    
    if not preferred_gpus:
        all_gpus_by_time = [(gpu_timeline[i], i) for i in range(len(gpu_timeline))]
        earliest_k_gpus = heapq.nsmallest(k, all_gpus_by_time)
        
        selected = [gpu_idx for time, gpu_idx in earliest_k_gpus]
        start_time = max(time for time, gpu_idx in earliest_k_gpus) if earliest_k_gpus else 0.0
        return selected, start_time

    
    preferred_candidates = []
    other_candidates = []
    
    preferred_set = set(preferred_gpus)
    for gpu_idx, free_time in enumerate(gpu_timeline):
        candidate = (free_time, gpu_idx)
        if gpu_idx in preferred_set:
            preferred_candidates.append(candidate)
        else:
            other_candidates.append(candidate)
            
    preferred_candidates.sort()
    other_candidates.sort()
    
    selected_gpus = []
    if len(preferred_candidates) >= k:
        selected_gpus = preferred_candidates[:k]
    else:
        selected_gpus.extend(preferred_candidates)
        needed = k - len(preferred_candidates)
        selected_gpus.extend(other_candidates[:needed])
        
    final_gpu_list = [gpu_idx for time, gpu_idx in selected_gpus]
    final_start_time = max(time for time, gpu_idx in selected_gpus) if selected_gpus else 0.0
    
    return final_gpu_list, final_start_time


def _get_smart_local_candidates(gpus_on_machine: List[int], k: int, gpu_timeline: List[float], topology_model: TopologyModel) -> List[Tuple]:
    """A few high-quality k-GPU groups from one machine:
    earliest-free, physically contiguous (low/high ids), NUMA-aligned."""
    if len(gpus_on_machine) < k: return []
    
    candidates = set()
    
    candidates.add(tuple(sorted(sorted(gpus_on_machine, key=lambda g: gpu_timeline[g])[:k])))
    
    sorted_by_id = sorted(gpus_on_machine)
    candidates.add(tuple(sorted_by_id[:k]))
    if len(sorted_by_id) > k: candidates.add(tuple(sorted_by_id[-k:]))

    numa0_gpus = [g for g in gpus_on_machine if topology_model._gpu_to_group_label(g) in ['A', 'B']]
    numa1_gpus = [g for g in gpus_on_machine if topology_model._gpu_to_group_label(g) in ['C', 'D']]
    if len(numa0_gpus) >= k: candidates.add(tuple(sorted(numa0_gpus)[:k]))
    if len(numa1_gpus) >= k: candidates.add(tuple(sorted(numa1_gpus)[:k]))
        
    return list(candidates)


def find_best_gpu_allocation_v5(
    gpu_timeline: List[float], k: int, base_duration: float,
    earliest_start_time: float, topology_model: 'TopologyModel',
    lookahead_threshold: float = 20.0, beam_width: int = 5,
    task_key: Tuple = None, log_file: IO = None
) -> Tuple[List[int], float, float, float]:
    """【V5 - Beam Search】"""
    if k == 0: return [], earliest_start_time, earliest_start_time, 1.0
    
    potential_start_times = sorted(list(set(gpu_timeline)))
    best_allocation = { "gpus": None, "start_time": float('inf'), "finish_time": float('inf'), "penalty": float('inf') }
    search_times = sorted(list(set([t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time])))
    search_times = search_times[:8]

    for t_start in search_times:
        if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold: break
        available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]
        if len(available_gpus) < k: continue

        all_candidate_gpus, machine_groups = [], defaultdict(list)
        for gpu in available_gpus: machine_groups[gpu // topology_model.machine_size].append(gpu)
        
        for _, gpus_on_machine in machine_groups.items():
            if len(gpus_on_machine) >= k:
                all_candidate_gpus.extend(list(itertools.combinations(gpus_on_machine, k)))
        
        split_plans = _generate_split_plans(k, machine_groups)
        MAX_COMBOS_PER_SPLIT = 100
        for plan in split_plans:
            machine_combo_iterators = []
            for machine_id, count in plan.items():
                combos = list(itertools.combinations(machine_groups[machine_id], count))
                if len(combos) > MAX_COMBOS_PER_SPLIT:
                    combos = random.sample(combos, MAX_COMBOS_PER_SPLIT)
                machine_combo_iterators.append(combos)
            if machine_combo_iterators:
                for cross_machine_combo_parts in itertools.product(*machine_combo_iterators):
                    final_combo = tuple(sorted([gpu for part in cross_machine_combo_parts for gpu in part]))
                    all_candidate_gpus.append(final_combo)
        
        scored_candidates = []
        for gpu_tuple in set(all_candidate_gpus):
            gpu_list = list(gpu_tuple)
            penalty = topology_model.predict_penalty(gpu_list)
            distance = topology_model.calculate_numa_distance(gpu_list)
            scored_candidates.append((penalty, distance, gpu_list))

        scored_candidates.sort(key=lambda x: (x[0], x[1]))
        top_k_candidates = scored_candidates[:beam_width]

        if log_file:
            log_file.write(f"--- Decision for Task {task_key} at t_start={t_start:.2f} ---\n")
            log_file.write(f"Found {len(scored_candidates)} candidates. Evaluating top {beam_width} (Beam Width).\n")
            log_file.write("Rank | Penalty  | NUMA Dist | GPU List               | Status\n")
            log_file.write("---- | -------- | --------- | ---------------------- | ------\n")
            for i, (penalty, distance, gpu_list) in enumerate(scored_candidates):
                status = "IN BEAM" if i < beam_width else "Pruned"
                if i < max(beam_width, 20):
                    log_file.write(f"{i+1:<4} | {penalty:<8.4f} | {distance:<9.1f} | {str(gpu_list):<22} | {status}\n")
            if len(scored_candidates) > 20:
                log_file.write("... (rest of the pruned candidates omitted for brevity) ...\n")
            log_file.write("\n")

        for penalty, _, gpu_list in top_k_candidates:
            actual_duration = base_duration * penalty
            finish_time = t_start + actual_duration
            if finish_time < best_allocation["finish_time"]:
                best_allocation.update({ "gpus": gpu_list, "start_time": t_start, "finish_time": finish_time, "penalty": penalty })
    
    if best_allocation["gpus"] is None:
        sorted_gpus_by_time = sorted(enumerate(gpu_timeline), key=lambda x: x[1])
        gpus_to_wait_for = [g[0] for g in sorted_gpus_by_time[:k]]
        forced_start_time = max(gpu_timeline[g] for g in gpus_to_wait_for)
        penalty = topology_model.predict_penalty(gpus_to_wait_for)
        actual_duration = base_duration * penalty
        finish_time = forced_start_time + actual_duration
        return gpus_to_wait_for, forced_start_time, finish_time, penalty

    return (best_allocation["gpus"], best_allocation["start_time"], best_allocation["finish_time"], best_allocation["penalty"])


def find_best_gpu_allocation_v6(
    gpu_timeline,
    k,
    base_duration,
    earliest_start_time,
    topology_model,
    lookahead_threshold=20.0,
    beam_width=5,
    task_key=None,
    log_file=None
):
    if k == 0:
        return [], earliest_start_time, earliest_start_time, 1.0

    # Truncate candidate start times (critical for large clusters)
    search_times = sorted(t for t in set(gpu_timeline) if t >= earliest_start_time)
    search_times = [earliest_start_time] + search_times[:6]

    best = {
        "gpus": None,
        "start": float("inf"),
        "finish": float("inf"),
        "penalty": float("inf"),
    }

    for t_start in search_times:
        if best["gpus"] and t_start > best["start"] + lookahead_threshold:
            break

        available = [i for i, t in enumerate(gpu_timeline) if t <= t_start]
        if len(available) < k:
            continue

        machine_groups = defaultdict(list)
        for g in available:
            machine_groups[g // topology_model.machine_size].append(g)

        # ---- cheap score ----
        def cheap_score(gpu_tuple):
            nodes = {g // topology_model.machine_size for g in gpu_tuple}
            span = gpu_tuple[-1] - gpu_tuple[0]
            return (len(nodes), span)

        split_plans = _generate_split_plans(k, machine_groups)

        for plan in split_plans:
            per_machine_candidates = []

            num_nodes = len(plan)
            limit = 3 if num_nodes >= 4 else 5 if num_nodes == 3 else 10

            valid = True
            for m_id, cnt in plan.items():
                local = _get_smart_local_candidates(
                    machine_groups[m_id],
                    cnt,
                    gpu_timeline,
                    topology_model,
                )
                if not local:
                    valid = False
                    break
                per_machine_candidates.append(local[:limit])

            if not valid:
                continue

            # =====================================================
            # Beam merge (replaces the full cartesian product)
            # =====================================================
            merged_candidates = beam_merge_candidates(
                per_machine_candidates,
                topology_model,
                beam_width=beam_width * 4,
                cheap_score_fn=cheap_score,
            )

            # =====================================================
            # Expensive topology scoring (merged)
            # =====================================================
            scored = []
            for gpu_tuple in merged_candidates:
                penalty, dist = topology_model.score_bitmask(gpu_tuple)
                scored.append((penalty, dist, list(gpu_tuple)))

            scored.sort(key=lambda x: (x[0], x[1]))
            topk = scored[:beam_width]

            for penalty, _, gpus in topk:
                finish = t_start + base_duration * penalty
                if finish < best["finish"]:
                    best.update(
                        gpus=gpus,
                        start=t_start,
                        finish=finish,
                        penalty=penalty,
                    )

    if best["gpus"] is None:
        idx = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])[:k]
        start = max(gpu_timeline[i] for i in idx)
        penalty, _ = topology_model.score_bitmask(tuple(idx))
        finish = start + base_duration * penalty
        return idx, start, finish, penalty

    return best["gpus"], best["start"], best["finish"], best["penalty"]


def beam_merge_candidates(
    per_machine_candidates,
    topology_model,
    beam_width,
    cheap_score_fn,
):
    """Beam-merge per-machine candidate tuples (sorted/truncated per
    machine) into cross-machine groups; returns <= ~beam_width results."""
    beam = [()]  # partial gpu tuples

    for machine_cands in per_machine_candidates:
        new_beam = []

        for partial in beam:
            for cand in machine_cands:
                merged = tuple(sorted(partial + cand))
                new_beam.append(merged)

        # ---- cheap prune ----
        new_beam.sort(key=cheap_score_fn)
        new_beam = new_beam[:beam_width * 2]

        beam = new_beam

    return beam[:beam_width]


def find_best_gpu_allocation_resource_only(
    gpu_timeline,
    k,
    base_duration,
    earliest_start_time,
):
    """
    Topology-agnostic placement:
    - GPU ids are meaningless
    - Only resource capacity matters
    """
    available_times = sorted(gpu_timeline)
    
    start = max(earliest_start_time, available_times[k - 1])
    finish = start + base_duration

    gpus = list(range(k))
    return gpus, start, finish, 1.0


# @profile
def find_best_gpu_allocation_v6_origin(
    gpu_timeline: List[float], k: int, base_duration: float,
    earliest_start_time: float, topology_model: 'TopologyModel',
    lookahead_threshold: float = 20.0, beam_width: int = 5,
    task_key: Tuple = None, log_file: IO = None
) -> Tuple[List[int], float, float, float]:
    """【V6 - Beam Search】"""
    if k == 0: return [], earliest_start_time, earliest_start_time, 1.0
    
    potential_start_times = sorted(list(set(gpu_timeline)))
    best_allocation = { "gpus": None, "start_time": float('inf'), "finish_time": float('inf'), "penalty": float('inf') }
    search_times = sorted(list(set([t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time])))

    for t_start in search_times:
        if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold: break
        available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]
        if len(available_gpus) < k: continue

# Candidate generation: exhaustively enumerating all k-GPU subsets is
        # infeasible, so build a small, high-quality pool instead.
        # 1) intra-node candidates: per machine, a few "smart" local groups
        #    (earliest-free / physically contiguous / NUMA-aligned) -- a
        #    single-node group pays no cross-node penalty (penalty = 1.0).
        smart_candidates, machine_groups = set(), defaultdict(list)
        for gpu in available_gpus: machine_groups[gpu // topology_model.machine_size].append(gpu)

        for _, gpus_on_machine in machine_groups.items():
            smart_candidates.update(_get_smart_local_candidates(gpus_on_machine, k, gpu_timeline, topology_model))

        # 2) cross-node candidates: enumerate machine split plans and take the
        #    cartesian product of each machine's top local groups.
        split_plans = _generate_split_plans(k, machine_groups)

        
        for plan in split_plans:

            # Adapt the per-machine contribution limit to the span of the
            # plan so the total number of combinations stays around ~100
            # regardless of how many nodes the group crosses:
            num_nodes_in_plan = len(plan)
            if num_nodes_in_plan >= 4:
                limit = 3   # >= 4 nodes: 3^4 = 81 combos
            elif num_nodes_in_plan == 3:
                limit = 5   # 3 nodes: 5^3 = 125 combos
            else:
                limit = 10  # <= 2 nodes: 10^2 = 100 combos
            machine_local_candidates = []
            valid_plan = True
            for machine_id, count in plan.items():
                local_candidates = _get_smart_local_candidates(machine_groups[machine_id], count, gpu_timeline, topology_model)
                if not local_candidates:
                    valid_plan = False
                    break
                machine_local_candidates.append(local_candidates[:limit])
            if not valid_plan: continue

            for cross_machine_parts in itertools.product(*machine_local_candidates):
                final_combo = tuple(sorted([gpu for part in cross_machine_parts for gpu in part]))
                smart_candidates.add(final_combo)
        
        if not smart_candidates: continue
        
        scored_candidates = []
        for gpu_tuple in smart_candidates:
            gpu_list = gpu_tuple
            penalty = topology_model.predict_penalty(gpu_list)
            distance = topology_model.calculate_numa_distance(gpu_list)
            gpu_list = list(gpu_tuple)
            scored_candidates.append((penalty, distance, gpu_list))
            

        scored_candidates.sort(key=lambda x: (x[0], x[1]))
        top_k_candidates = scored_candidates[:beam_width]

        if log_file:
            log_file.write(f"--- Decision for Task {task_key} at t_start={t_start:.2f} ---\n")
            log_file.write(f"Found {len(scored_candidates)} candidates. Evaluating top {beam_width} (Beam Width).\n")
            log_file.write("Rank | Penalty  | NUMA Dist | GPU List               | Status\n")
            log_file.write("---- | -------- | --------- | ---------------------- | ------\n")
            for i, (penalty, distance, gpu_list) in enumerate(scored_candidates):
                status = "IN BEAM" if i < beam_width else "Pruned"
                if i < max(beam_width, 20):
                    log_file.write(f"{i+1:<4} | {penalty:<8.4f} | {distance:<9.1f} | {str(gpu_list):<22} | {status}\n")
            if len(scored_candidates) > 20:
                log_file.write("... (rest of the pruned candidates omitted for brevity) ...\n")
            log_file.write("\n")

        for penalty, _, gpu_list in top_k_candidates:
            actual_duration = base_duration * penalty
            finish_time = t_start + actual_duration
            if finish_time < best_allocation["finish_time"]:
                best_allocation.update({ "gpus": gpu_list, "start_time": t_start, "finish_time": finish_time, "penalty": penalty })
    
    if best_allocation["gpus"] is None:
        sorted_gpus_by_time = sorted(enumerate(gpu_timeline), key=lambda x: x[1])
        gpus_to_wait_for = [g[0] for g in sorted_gpus_by_time[:k]]
        forced_start_time = max(gpu_timeline[g] for g in gpus_to_wait_for)
        penalty = topology_model.predict_penalty(gpus_to_wait_for)
        actual_duration = base_duration * penalty
        finish_time = forced_start_time + actual_duration
        return gpus_to_wait_for, forced_start_time, finish_time, penalty

    return (best_allocation["gpus"], best_allocation["start_time"], best_allocation["finish_time"], best_allocation["penalty"])


def find_buddy_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    """Buddy-system allocation: k = 2^n GPUs on an aligned contiguous block."""
    assert (k & (k - 1)) == 0, "k must be a power of two"

    n = len(gpu_timeline)
    block_size = k
    for i in range(0, n, block_size):
        if i + block_size > n:
            continue
        block = list(range(i, i + block_size))
        start_time = max(gpu_timeline[g] for g in block)
        yield block, start_time


def heuristic(finished, tasks, total_gpus):
    remain = 0
    # print(f"{tasks}")
    for i in range(len(tasks)):
        if f"{i}_DIT" not in finished:
            remain += min(t[1] for t in tasks[i]["dit"])
        if f"{i}_VAE" not in finished:
            remain += min(t[1] for t in tasks[i]["vae"])
    return remain / total_gpus


def greedy_schedule(tasks: List[Dict], total_gpus: int = 8):
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}

    for i in range(len(tasks)):
        dataset_id = tasks[i]["dataset_id"]

        selected = False
        for k, t in sorted(tasks[i]["vae"], key=lambda x: x[1]):
            candidate_blocks = list(find_earliest_gpus(gpu_timeline, k))
            if not candidate_blocks:
                continue
            gpus, start = min(candidate_blocks, key=lambda x: x[1])
            end = start + t
            for g in gpus:
                gpu_timeline[g] = end
            trace.append((i, "VAE", k, t, start, end, dataset_id, gpus))
            stage_end_time[f"{i}_VAE"] = end
            selected = True
            break
        if not selected:
            raise RuntimeError(f"VAE stage: cannot allocate {k} GPUs for task {i}")

        selected = False
        for k, t in sorted(tasks[i]["dit"], key=lambda x: x[1]):
            candidate_blocks = list(find_buddy_gpus(gpu_timeline, k))
            if not candidate_blocks:
                continue
            gpus, tentative_start = min(candidate_blocks, key=lambda x: x[1])
            start = max(tentative_start, stage_end_time[f"{i}_VAE"])
            end = start + t
            for g in gpus:
                gpu_timeline[g] = end
            trace.append((i, "DIT", k, t, start, end, dataset_id, gpus))
            stage_end_time[f"{i}_DIT"] = end
            selected = True
            break
        if not selected:
            raise RuntimeError(f"DIT stage: cannot allocate {k} GPUs for task {i}")

    makespan = max(gpu_timeline)
    return trace, makespan


def a_star_schedule(tasks: List[Dict], total_gpus: int = 32):
    initial = State(set(), [0.0] * total_gpus, 0.0, [], dict())
    heap = [(initial.total_time + heuristic(set(), tasks, total_gpus), initial)]

    while heap:
        _, state = heapq.heappop(heap)
        if len(state.finished) == 2 * len(tasks):
            return state

        for i in range(len(tasks)):
            dit_key, vae_key = f"{i}_DIT", f"{i}_VAE"

            if vae_key not in state.finished:
                for k, t in tasks[i]["vae"]:
                    if total_gpus >= k:
                        gpus, start = find_earliest_gpus(state.gpu_timeline, k)


                        end = start + t
                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end
                        new_finished = state.finished | {vae_key}
                        new_trace = state.trace + [
                            (i, "VAE", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[vae_key] = end
                        new_state = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )
                        heapq.heappush(
                            heap,
                            (
                                new_state.total_time
                                + heuristic(new_finished, tasks, total_gpus),
                                new_state,
                            ),
                        )

            elif dit_key not in state.finished:
                for k, t in tasks[i]["dit"]:
                    if total_gpus >= k:
                        gpus, gpu_ready = find_earliest_gpus(state.gpu_timeline, k)
                        vae_end = state.stage_end_time[vae_key]
                        start = max(gpu_ready, vae_end)
                        end = start + t
                        new_gpu = state.gpu_timeline[:]
                        for g in gpus:
                            new_gpu[g] = end
                        new_finished = state.finished | {dit_key}
                        new_trace = state.trace + [
                            (i, "DIT", k, t, start, end, tasks[i]["dataset_id"], gpus)
                        ]
                        new_end_times = state.stage_end_time.copy()
                        new_end_times[dit_key] = end
                        new_state = State(
                            new_finished,
                            new_gpu,
                            max(end, state.total_time),
                            new_trace,
                            new_end_times,
                        )
                        heapq.heappush(
                            heap,
                            (
                                new_state.total_time
                                + heuristic(new_finished, tasks, total_gpus),
                                new_state,
                            ),
                        )


def plot_schedule_timeline(trace, save_path="generated_schedule_timeline.png"):
    """Render the schedule trace as a per-task timeline figure."""
    df = pd.DataFrame(
        trace, columns=["Task", "Stage", "GPUs", "Time", "Start", "End", "DatasetID"]
    )

    df["TaskLabel"] = df.apply(
        lambda row: f"Task {row['Task']} [D{row['DatasetID']}]", axis=1
    )
    unique_labels = df["TaskLabel"].unique()[::-1]

    plt.figure(figsize=(12, max(6, len(unique_labels) * 0.6)))
    colors = {"DIT": "skyblue", "VAE": "salmon"}

    for idx, row in df.iterrows():
        plt.barh(
            y=row["TaskLabel"],
            width=row["Time"],
            left=row["Start"],
            color=colors[row["Stage"]],
            edgecolor="black",
            label=row["Stage"] if idx < 2 else "",
        )
        plt.text(
            row["Start"] + row["Time"] / 2,
            row["TaskLabel"],
            f'{row["Stage"]}-{row["GPUs"]}G',
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Task")
    plt.title("Task Scheduling Timeline (with Dataset ID)")
    plt.legend(loc="upper right")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
    """Render the schedule trace as a per-GPU timeline (one row per GPU)."""
    records = []
    for task_id, stage, k, t, start, end, dataset_id, gpu_list in trace:
        for g in gpu_list:
            records.append(
                {
                    "GPU": g,
                    "Task": task_id,
                    "Stage": stage,
                    "Start": start,
                    "End": end,
                    "Time": t,
                    "Label": f"{stage}-T{task_id}[D{dataset_id}]",
                }
            )

    df = pd.DataFrame(records)
    df["GPU"] = df["GPU"].astype(int)

    plt.figure(figsize=(12, max(6, total_gpus * 0.5)))
    colors = {"DIT": "skyblue", "VAE": "salmon"}

    for _, row in df.iterrows():
        plt.barh(
            y=row["GPU"],
            width=row["Time"],
            left=row["Start"],
            color=colors.get(row["Stage"], "gray"),
            edgecolor="black",
        )
        plt.text(
            row["Start"] + row["Time"] / 2,
            row["GPU"],
            row["Label"],
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    plt.yticks(range(total_gpus), [f"GPU {i}" for i in range(total_gpus)])
    plt.xlabel("Time (s)")
    plt.ylabel("GPU ID")
    plt.title("GPU Usage Timeline by Rank")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# =============================================================================


def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    """Critical-path length from this task to the sink (list-scheduling priority)."""
    if not hasattr(get_critical_path_priority, "cache"):
        get_critical_path_priority.cache = {}
    if task_key in get_critical_path_priority.cache:
        return get_critical_path_priority.cache[task_key]

    successors = [t for t, preds in predecessors.items() if task_key in preds]

    if not successors:
        return proc_times[task_key]

    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times) for succ in successors)
    result = proc_times[task_key] + max_succ_path
    get_critical_path_priority.cache[task_key] = result
    return result


def generate_trace_from_chromosome_v2(
    tasks_input: List[Dict],
    chromosome: List[int],
    total_gpus: int,
    topology_model: TopologyModel,
    log_file: IO = None,
    topology_aware: bool = True,
) -> Tuple[List[Tuple], float]:
    """Decode a chromosome into a trace; topology_aware=True uses the
    penalty-driven placement (intra-only), False uses resource-only
    placement (wo_mapper)."""
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}

    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')

        sp_vae = chromosome[chromosome_idx]
        sp_dit = chromosome[chromosome_idx + 1]

        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]

        sp_map[key_vae] = sp_vae
        sp_map[key_dit] = sp_dit
        predecessors[key_dit] = [key_vae]

        chromosome_idx += 2

    # 🔧 O(1) dataset lookup
    taskid_to_dataset = {t['task_id']: t['dataset_id'] for t in tasks_input}

    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()

    get_critical_path_priority.cache = {}
    task_list_internal.sort(
        key=lambda t: get_critical_path_priority(t, predecessors, proc_times),
        reverse=True
    )

    while len(completed_tasks) < len(proc_times):
        scheduled = False

        for task_key in list(task_list_internal):
            if task_key in completed_tasks:
                continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue

            k = sp_map[task_key]
            base_duration = proc_times[task_key]
            dep_finish_time = max(
                stage_end_time.get(dep, 0) for dep in deps
            ) if deps else 0
            if topology_aware:
                gpus, start, end, penalty = find_best_gpu_allocation_v6(
                    gpu_timeline,
                    k,
                    base_duration,
                    earliest_start_time=dep_finish_time,
                    topology_model=topology_model,
                    lookahead_threshold=3,
                    beam_width=5,
                    task_key=task_key,
                    log_file=log_file
                )
            else:
                gpus, start, end, penalty = find_best_gpu_allocation_resource_only(
                    gpu_timeline,
                    k,
                    base_duration,
                    earliest_start_time=dep_finish_time,
                )
            if gpus is None:
                continue

            for g in gpus:
                gpu_timeline[g] = end

            task_id, stage = task_key
            dataset_id = taskid_to_dataset[task_id]

            trace.append(
                (task_id, stage, k, end - start, start, end, dataset_id, gpus, penalty)
            )

            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            scheduled = True
            break

        if not scheduled:
            raise RuntimeError("Scheduling deadlock: check resources and dependencies.")

    return trace, max(gpu_timeline) if gpu_timeline else 0.0


# =============================================================================
# @profile
def generate_trace_from_chromosome_v2_origin(tasks_input: List[Dict], chromosome: List[int], total_gpus: int, 
                                     topology_model: TopologyModel, log_file: IO = None) -> Tuple[List[Tuple], float]:
    """Decode a chromosome (per-task SP choices) into a trace via
    topology-aware list scheduling; durations are base * comm-penalty."""
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}
    
    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')
        sp_vae = chromosome[chromosome_idx]; sp_dit = chromosome[chromosome_idx+1]
        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
        sp_map[key_vae], sp_map[key_dit] = sp_vae, sp_dit
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2
    # ...

    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks: continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue
            
            k = sp_map[task_key]
            base_duration = proc_times[task_key]
            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0

            gpus, start, end, penalty = find_best_gpu_allocation_v6_origin(
                gpu_timeline,
                k,
                base_duration,
                earliest_start_time=dep_finish_time,
                topology_model=topology_model,
                lookahead_threshold=3,
                beam_width=5,
                task_key=task_key,
                log_file=log_file
            )
            
            if gpus is None:
                continue

            actual_duration = end - start
            for g in gpus:
                gpu_timeline[g] = end
            
            task_id, stage = task_key
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage, k, actual_duration, start, end, dataset_id, gpus, penalty))
            
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            scheduled_this_loop = True
            break
        
        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("Scheduling deadlock: check resources and dependencies.")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


def generate_trace_from_chromosome_v1_5_inter_aware(
    tasks_input: List[Dict], 
    chromosome: List[int], 
    total_gpus: int
) -> Tuple[List[Tuple], float]:
    """Decode a chromosome into a trace with inter-cascade affinity:
    each DIT is steered onto its own VAE's GPUs (no topology penalty)."""
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}
    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')
        sp_vae = chromosome[chromosome_idx]; sp_dit = chromosome[chromosome_idx+1]
        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
        sp_map[key_vae], sp_map[key_dit] = sp_vae, sp_dit
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2
        
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    vae_gpu_placements = {}
    # ===============================================

    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks: continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue
            
            k = sp_map[task_key]
            
            task_id, stage = task_key
            preferred_gpus_for_dit = None
            
            if stage == 'DIT' and task_id in vae_gpu_placements:
                preferred_gpus_for_dit = vae_gpu_placements[task_id]

            gpus, start_time_gpu = find_gpus_with_cascade_awareness(
                gpu_timeline, k, preferred_gpus=preferred_gpus_for_dit
            )
            # ===================================================

            if gpus is None: continue

            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0
            
            start = max(start_time_gpu, dep_finish_time)
            duration = proc_times[task_key]
            end = start + duration

            for g in gpus: gpu_timeline[g] = end
            
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage, k, duration, start, end, dataset_id, gpus))
            
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            
            if stage == 'VAE':
                vae_gpu_placements[task_id] = gpus
            # ===============================================

            scheduled_this_loop = True
            break
        
        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("Scheduling deadlock: check resources and dependencies.")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


def generate_trace_from_chromosome_v1(tasks_input: List[Dict], chromosome: List[int], total_gpus: int) -> Tuple[List[Tuple], float]:
    """Decode a chromosome into a trace via plain list scheduling
    (earliest-free placement, no topology awareness)."""
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}

    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae = (task_id, 'VAE')
        key_dit = (task_id, 'DIT')
        
        sp_vae = chromosome[chromosome_idx]
        task_list_internal.append(key_vae)
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        sp_map[key_vae] = sp_vae
        chromosome_idx += 1
        
        sp_dit = chromosome[chromosome_idx]
        task_list_internal.append(key_dit)
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
        sp_map[key_dit] = sp_dit
        chromosome_idx += 1
        
        predecessors[key_dit] = [key_vae]

    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks: continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue
            
            k = sp_map[task_key]
            gpus, start_time_gpu = find_earliest_gpus(gpu_timeline, k)
            
            if gpus is None: continue

            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0
            
            start = max(start_time_gpu, dep_finish_time)
            duration = proc_times[task_key]
            end = start + duration

            for g in gpus: gpu_timeline[g] = end
            
            task_id, stage = task_key
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage, k, duration, start, end, dataset_id, gpus))
            
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            scheduled_this_loop = True
        
        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("Scheduling deadlock: check resources and dependencies.")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan

def tournament_select(pop, fitnesses, k=3):
    cand = random.sample(range(len(pop)), k)
    best = min(cand, key=lambda i: fitnesses[i])
    return pop[best]

def build_sp_options(tasks_input, total_gpus):
    sp_options = []
    for task_def in tasks_input:
        vae_opts = [sp for sp, _ in task_def["vae"] if sp <= total_gpus]
        dit_opts = [sp for sp, _ in task_def["dit"] if sp <= total_gpus]
        if not vae_opts or not dit_opts:
            raise ValueError(f"Task {task_def['task_id']} has no feasible SP <= {total_gpus}")
        sp_options.append(vae_opts)
        sp_options.append(dit_opts)
    return sp_options

def genetic_algorithm_schedule_v1_improved(
    tasks_input,
    total_gpus,
    population_size=60,
    num_generations=80,
    mutation_rate=0.10,
    elitism_size=2,
    local_k=5,
    local_tries=3,
    progress_recorder: list = None,
):
    sp_options_per_gene = build_sp_options(tasks_input, total_gpus)

    keys = []
    dur_map = {}
    deps = {}
    taskid_to_dataset = {t["task_id"]: t.get("dataset_id", "") for t in tasks_input}

    # chromosome: [vae_sp0, dit_sp0, vae_sp1, dit_sp1, ...]
    gene_to_taskkey = []
    for t in tasks_input:
        tid = t["task_id"]
        gene_to_taskkey.append((tid, "VAE"))
        gene_to_taskkey.append((tid, "DIT"))
        deps[(tid, "DIT")] = [(tid, "VAE")]
    task_by_id = {t["task_id"]: t for t in tasks_input}
    def eval_chromosome(chrom):
        k_map = {}
        dur_map_local = {}
        for gi, task_key in enumerate(gene_to_taskkey):
            sp = chrom[gi]
            k_map[task_key] = sp
            tid, stage = task_key
            task = task_by_id[tid]
            table = dict(task["vae"] if stage == "VAE" else task["dit"])
            dur_map_local[task_key] = table[sp]

        all_tasks = gene_to_taskkey
        return schedule_makespan_heap_ready(
            all_tasks=all_tasks,
            k_map=k_map,
            dur_map=dur_map_local,
            deps=deps,
            total_gpus=total_gpus,
            pick_rule="kxdur",
        )

    # --- init population ---
    population = [[random.choice(opts) for opts in sp_options_per_gene] for _ in range(population_size)]
    best_chrom, best_fit = None, float("inf")

    for gen in range(num_generations):
        fitness = [eval_chromosome(ch) for ch in population]

        i_best = min(range(population_size), key=lambda i: fitness[i])
        if fitness[i_best] < best_fit:
            best_fit = fitness[i_best]
            best_chrom = population[i_best][:]

        # local search on top-K
        top_idx = sorted(range(population_size), key=lambda i: fitness[i])[:local_k]
        for i in top_idx:
            improved, val = local_improve(population[i], sp_options_per_gene, eval_chromosome, tries=local_tries)
            population[i] = improved
            fitness[i] = val
            if val < best_fit:
                best_fit, best_chrom = val, improved[:]

        # elitism
        order = sorted(range(population_size), key=lambda i: fitness[i])
        new_pop = [population[i][:] for i in order[:elitism_size]]

        # reproduction
        while len(new_pop) < population_size:
            p1 = tournament_select(population, fitness, k=3)
            p2 = tournament_select(population, fitness, k=3)
            point = random.randint(1, len(p1)-1)
            c = p1[:point] + p2[point:]
            # mutation
            for j in range(len(c)):
                if random.random() < mutation_rate:
                    c[j] = random.choice(sp_options_per_gene[j])
            new_pop.append(c)

        population = new_pop

    final_trace, final_makespan = generate_trace_from_chromosome_v1(tasks_input, best_chrom, total_gpus)
    return final_trace, final_makespan


def genetic_algorithm_schedule_v1(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2, log_file_path: str = None):
    """GA scheduler -- NO-mapper baseline (wo_mapper ablation):
    earliest-free GPU placement, no topology awareness."""
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        fitnesses = {}


        for i, chrom in enumerate(population):
            _, makespan = generate_trace_from_chromosome_v1(tasks_input, chrom, total_gpus)
            fitnesses[i] = makespan

        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < population_size:
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            def mutate(chromosome):
                mutated_chromosome = list(chromosome)
                for i in range(len(mutated_chromosome)):
                    if random.random() < mutation_rate:
                        mutated_chromosome[i] = random.choice(sp_options_per_task[i])
                return mutated_chromosome

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        
        population = new_population

    print(f"\n[GA] Evolution complete. Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration (chromosome): {best_overall_chromosome}")

    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v1(tasks_input, best_overall_chromosome, total_gpus)
    
    return final_trace, final_makespan


def genetic_algorithm_schedule_v1_5(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2, log_file_path: str = None):
    """GA scheduler -- inter-cascade affinity ONLY (wo_intra ablation):
    a DIT task prefers its own VAE's GPUs; no topology penalty."""
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        fitnesses = {}
        for i, chrom in enumerate(population):
            _, makespan = generate_trace_from_chromosome_v1_5_inter_aware(tasks_input, chrom, total_gpus)
            fitnesses[i] = makespan

        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < population_size:
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            def mutate(chromosome):
                mutated_chromosome = list(chromosome)
                for i in range(len(mutated_chromosome)):
                    if random.random() < mutation_rate:
                        mutated_chromosome[i] = random.choice(sp_options_per_task[i])
                return mutated_chromosome

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        
        population = new_population

    print(f"\n[GA] Evolution complete. Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration (chromosome): {best_overall_chromosome}")

    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v1_5_inter_aware(tasks_input, best_overall_chromosome, total_gpus)
    
    return final_trace, final_makespan


def evaluate_chromosome_wrapper(chromosome, tasks_input, total_gpus, topology_model, use_topology):
    """Per-chromosome evaluation, pickled into parallel worker processes."""
    _, makespan = generate_trace_from_chromosome_v2(
        tasks_input, 
        chromosome, 
        total_gpus, 
        topology_model=topology_model, 
        log_file=None,
        topology_aware=use_topology
    )
    return makespan


def evaluate_chromosome_wrapper_origin(chromosome, tasks_input, total_gpus, topology_model):
    """Per-chromosome evaluation, pickled into parallel worker processes."""
    _, makespan = generate_trace_from_chromosome_v2_origin(
        tasks_input, 
        chromosome, 
        total_gpus, 
        topology_model=topology_model, 
        log_file=None,
    )
    return makespan

def genetic_algorithm_schedule_v2(
    tasks_input: List[Dict],
    total_gpus: int,
    population_size=50,
    num_generations=50,
    mutation_rate=0.1,
    elitism_size=2,
    log_file_path: str = None,
    machine_size: int = None,
    progress_recorder: list = None,   
    use_topology: bool = True,
):
    """GA scheduler -- intra-cascade topology placement ONLY (wo_inter
    ablation); use_topology=False degrades to resource-only placement."""
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    if machine_size is None:
        machine_size = total_gpus
    try:
        topology_model = TopologyModel(params_file='fit_results.json', machine_size=machine_size, enable_topology=use_topology)
    except Exception as e:
        print(f"❌ Error initializing TopologyModel: {e}")
        print("Falling back to topology-agnostic scheduling (all penalties = 1.0).")
        topology_model = type('DummyModel', (), {'predict_penalty': lambda self, gl: 1.0})()
    log_file = open(log_file_path, 'w') if log_file_path else None
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')
    env_cores = os.environ.get("GA_NUM_CORES")
    if env_cores is not None:
        num_cores_to_use = min(int(env_cores), population_size)
    else:
        num_cores_to_use = min(os.cpu_count() or 64, population_size)
    # num_cores_to_use = 1
    print(f"🚀 Activating Parallel Evaluation on {num_cores_to_use} cores!")
    import time

    ga_start_time = time.time()

    if progress_recorder is not None:
        progress_recorder.clear()
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        makespan_results = Parallel(n_jobs=num_cores_to_use, 
                                    backend='loky')(
                    delayed(evaluate_chromosome_wrapper)(
                        chrom, tasks_input, total_gpus, topology_model, use_topology
                    ) for chrom in population
                )
        
        fitnesses = {i: ms for i, ms in enumerate(makespan_results)}
        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            if progress_recorder is not None:
                progress_recorder.append({
                    "generation": gen + 1,
                    "best_makespan": best_overall_fitness,
                    "elapsed_plan_time": time.time() - ga_start_time,
                })
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < population_size:
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            def mutate(chromosome):
                mutated_chromosome = list(chromosome)
                for i in range(len(mutated_chromosome)):
                    if random.random() < mutation_rate:
                        mutated_chromosome[i] = random.choice(sp_options_per_task[i])
                return mutated_chromosome

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        
        population = new_population

    print(f"\n[GA] Evolution complete. Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration (chromosome): {best_overall_chromosome}")

    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v2(tasks_input, best_overall_chromosome, total_gpus, topology_model=topology_model)
    
    final_trace_formatted = [item[:-1] for item in final_trace]
    return final_trace_formatted, final_makespan


def genetic_algorithm_schedule_v2_origin(
    tasks_input: List[Dict],
    total_gpus: int,
    population_size=50,
    num_generations=50,
    mutation_rate=0.1,
    elitism_size=2,
    log_file_path: str = None,
    machine_size: int = None,
    progress_recorder: list = None,   
):
    """Production GA scheduler -- intra-cascade topology-aware placement
    (used for the main experiments; = wo_inter w.r.t. the full joint mapper)."""
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    if machine_size is None:
        machine_size = total_gpus
    try:
        topology_model = TopologyModel(params_file='fit_results.json', machine_size=machine_size, enable_topology=True)
    except Exception as e:
        print(f"❌ Error initializing TopologyModel: {e}")
        print("Falling back to topology-agnostic scheduling (all penalties = 1.0).")
        topology_model = type('DummyModel', (), {'predict_penalty': lambda self, gl: 1.0})()
    log_file = open(log_file_path, 'w') if log_file_path else None
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')
    env_cores = os.environ.get("GA_NUM_CORES")
    if env_cores is not None:
        num_cores_to_use = min(int(env_cores), population_size)
    else:
        num_cores_to_use = min(os.cpu_count() or 64, population_size)
    # num_cores_to_use = 1
    print(f"🚀 Activating Parallel Evaluation on {num_cores_to_use} cores!")
    import time

    ga_start_time = time.time()

    if progress_recorder is not None:
        progress_recorder.clear()
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        makespan_results = Parallel(n_jobs=num_cores_to_use, 
                                    backend='loky')(
                    delayed(evaluate_chromosome_wrapper_origin)(
                        chrom, tasks_input, total_gpus, topology_model
                    ) for chrom in population
                )
        
        fitnesses = {i: ms for i, ms in enumerate(makespan_results)}
        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            if progress_recorder is not None:
                progress_recorder.append({
                    "generation": gen + 1,
                    "best_makespan": best_overall_fitness,
                    "elapsed_plan_time": time.time() - ga_start_time,
                })
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < population_size:
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            def mutate(chromosome):
                mutated_chromosome = list(chromosome)
                for i in range(len(mutated_chromosome)):
                    if random.random() < mutation_rate:
                        mutated_chromosome[i] = random.choice(sp_options_per_task[i])
                return mutated_chromosome

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        
        population = new_population

    print(f"\n[GA] Evolution complete. Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration (chromosome): {best_overall_chromosome}")

    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v2(tasks_input, best_overall_chromosome, total_gpus, topology_model=topology_model)

    final_trace_formatted = [item[:-1] for item in final_trace]
    return final_trace_formatted, final_makespan


# ============================================================================
# V3 -- Joint Intra+Inter Mapper (the FULL mapper of Table 3)
#
# v2_origin/v6_origin only does intra-cascade topology placement; v1_5 only
# does inter-cascade affinity. This version folds both into one objective on
# top of the v6_origin list-scheduling + beam-placement skeleton:
#
#   finish(G) = max(t_gpu_free(G), dep_end + inter_delay(G_src -> G))
#
# - intra_penalty: the production TopologyModel.predict_penalty (cross-node
#   1 + A*exp(-B*overlap), intra-node 1.0), multiplied into compute time;
# - inter_delay: data-movement latency on the VAE->DIT dependency edge
#   (full/partial/no-overlap tiers): 0 when dst is inside src (runtime
#   zero-copy cache hit), small NVLink cost for same-node misses, larger NIC
#   cost for cross-node misses; magnitude scales with the producer's actual
#   duration (activation size tracks compute) instead of absolute seconds;
# - candidate injection: besides v6_origin's smart intra candidates, inject
#   affinity candidates derived from the producer placement (exact reuse /
#   best subset / same-node fill) so the zero-transfer option is always in
#   the pool and the unified objective decides.
#
# Each placement explicitly trades queueing for the source GPUs vs transfer
# cost vs intra penalty; per-step complexity matches v6_origin.
# ============================================================================

# inter transfer-delay coefficients (relative to the producer's duration;
# defaults encode NVLink << NIC)
INTER_TRANSFER_LOCAL_FRAC = 0.02   # all misses on src's node(s): NVLink broadcast
INTER_TRANSFER_CROSS_FRAC = 0.08   # cross-node misses: NIC transfer


def _estimate_inter_transfer_delay(src_gpus, dst_gpus, producer_duration, machine_size):
    """Estimated inter-cascade transfer delay, added on the VAE->DIT edge."""
    if not src_gpus or producer_duration <= 0:
        return 0.0
    src = set(src_gpus)
    miss = [g for g in dst_gpus if g not in src]
    if not miss:
        return 0.0                       # full overlap: data already resident
    src_nodes = {g // machine_size for g in src}
    cross = any((g // machine_size) not in src_nodes for g in miss)
    frac = INTER_TRANSFER_CROSS_FRAC if cross else INTER_TRANSFER_LOCAL_FRAC
    # the larger the miss fraction, the more data must be moved
    return producer_duration * frac * (len(miss) / max(len(dst_gpus), 1))


def _inter_affinity_candidates(src_gpus, k, gpu_timeline, machine_size):
    """Affinity candidates derived from the producer (VAE) placement:
    - |src| == k: reuse the exact group (zero transfer);
    - |src| >  k: earliest-free k-subset + any single-node k-subset (penalty 1.0);
    - |src| <  k: the group filled with earliest-free same-node GPUs first.
    """
    cands = set()
    if not src_gpus:
        return []
    if len(src_gpus) == k:
        cands.add(tuple(sorted(src_gpus)))
    elif len(src_gpus) > k:
        by_free = sorted(src_gpus, key=lambda g: gpu_timeline[g])
        cands.add(tuple(sorted(by_free[:k])))
        by_node = defaultdict(list)
        for g in src_gpus:
            by_node[g // machine_size].append(g)
        for node_gpus in by_node.values():
            if len(node_gpus) >= k:
                cands.add(tuple(sorted(sorted(node_gpus, key=lambda g: gpu_timeline[g])[:k])))
    else:
        need = k - len(src_gpus)
        src_set = set(src_gpus)
        src_nodes = {g // machine_size for g in src_gpus}
        others = [g for g in range(len(gpu_timeline)) if g not in src_set]
        same_node = sorted([g for g in others if g // machine_size in src_nodes],
                           key=lambda g: gpu_timeline[g])
        cross_node = sorted([g for g in others if g // machine_size not in src_nodes],
                            key=lambda g: gpu_timeline[g])
        fill = (same_node + cross_node)[:need]
        if len(fill) == need:
            cands.add(tuple(sorted(list(src_gpus) + fill)))
    return list(cands)


def find_best_gpu_allocation_v7_joint(
    gpu_timeline: List[float], k: int, base_duration: float,
    earliest_start_time: float, topology_model: 'TopologyModel',
    producer_gpus: List[int] = None, producer_duration: float = 0.0,
    lookahead_threshold: float = 20.0, beam_width: int = 5,
    task_key: Tuple = None, log_file: IO = None
) -> Tuple[List[int], float, float, float]:
    """V7 joint beam search = v6_origin intra placement + inter transfer
    cost in one objective. `earliest_start_time` is the RAW dependency end;
    each candidate's effective ready time = dep_end + its own inter_delay."""
    if k == 0:
        return [], earliest_start_time, earliest_start_time, 1.0

    machine_size = topology_model.machine_size

    def _evaluate(gpu_tuple, t_floor):
        """Unified objective -> (finish, start, penalty, delay); t_floor is
        the availability floor of the candidate's member GPUs."""
        penalty = topology_model.predict_penalty(gpu_tuple)
        delay = _estimate_inter_transfer_delay(
            producer_gpus, gpu_tuple, producer_duration, machine_size)
        start = max(t_floor, earliest_start_time + delay)
        finish = start + base_duration * penalty
        return finish, start, penalty, delay

    best = {"gpus": None, "start": float('inf'), "finish": float('inf'), "penalty": float('inf')}

    # --- Phase 1: injected inter-affinity candidates (may wait for src GPUs) ---
    if producer_gpus:
        for cand in _inter_affinity_candidates(producer_gpus, k, gpu_timeline, machine_size):
            t_floor = max(gpu_timeline[g] for g in cand)
            finish, start, penalty, delay = _evaluate(cand, t_floor)
            if finish < best["finish"]:
                best.update(gpus=list(cand), start=start, finish=finish, penalty=penalty)

    # --- Phase 2: v6_origin-style start-time sweep with smart intra candidates ---
    potential_start_times = sorted(set(gpu_timeline))
    search_times = sorted(set([t for t in potential_start_times if t >= earliest_start_time]
                              + [earliest_start_time]))
    # Truncate the sweep (v6 uses 6; 12 is more conservative here): injected
    # candidates can push best.start later and delay the lookahead break, so
    # truncation bounds per-step complexity on 32/64-GPU clusters.
    search_times = search_times[:12]
    for t_start in search_times:
        if best["gpus"] and t_start > best["start"] + lookahead_threshold:
            break
        available_gpus = [i for i, free in enumerate(gpu_timeline) if free <= t_start]
        if len(available_gpus) < k:
            continue

        smart_candidates, machine_groups = set(), defaultdict(list)
        for gpu in available_gpus:
            machine_groups[gpu // machine_size].append(gpu)
        for _, gpus_on_machine in machine_groups.items():
            smart_candidates.update(_get_smart_local_candidates(
                gpus_on_machine, k, gpu_timeline, topology_model))
        split_plans = _generate_split_plans(k, machine_groups)
        for plan in split_plans:
            num_nodes_in_plan = len(plan)
            limit = 3 if num_nodes_in_plan >= 4 else 5 if num_nodes_in_plan == 3 else 10
            machine_local_candidates, valid_plan = [], True
            for machine_id, count in plan.items():
                local_candidates = _get_smart_local_candidates(
                    machine_groups[machine_id], count, gpu_timeline, topology_model)
                if not local_candidates:
                    valid_plan = False
                    break
                machine_local_candidates.append(local_candidates[:limit])
            if not valid_plan:
                continue
            for cross_machine_parts in itertools.product(*machine_local_candidates):
                smart_candidates.add(tuple(sorted(
                    gpu for part in cross_machine_parts for gpu in part)))
        if not smart_candidates:
            continue

        for gpu_tuple in smart_candidates:
            finish, start, penalty, delay = _evaluate(gpu_tuple, t_start)
            if finish < best["finish"]:
                best.update(gpus=list(gpu_tuple), start=start, finish=finish, penalty=penalty)

    # --- Fallback: wait for the k earliest-free GPUs (same as v6_origin) ---
    if best["gpus"] is None:
        idx = [g for g, _ in sorted(enumerate(gpu_timeline), key=lambda x: x[1])[:k]]
        t_floor = max(gpu_timeline[g] for g in idx)
        finish, start, penalty, _ = _evaluate(tuple(idx), t_floor)
        return idx, start, finish, penalty

    return best["gpus"], best["start"], best["finish"], best["penalty"]


def generate_trace_from_chromosome_v3_joint(
    tasks_input: List[Dict], chromosome: List[int], total_gpus: int,
    topology_model: 'TopologyModel', log_file: IO = None
) -> Tuple[List[Tuple], float]:
    """v2_origin skeleton with v7_joint placement: each VAE's actual
    placement/duration feeds its DIT's inter-transfer objective."""
    task_list_internal, proc_times, sp_map, predecessors = [], {}, {}, {}
    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')
        sp_vae = chromosome[chromosome_idx]; sp_dit = chromosome[chromosome_idx + 1]
        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
        sp_map[key_vae], sp_map[key_dit] = sp_vae, sp_dit
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2

    gpu_timeline = [0.0] * total_gpus
    trace, stage_end_time, completed_tasks = [], {}, set()
    vae_placements = {}      # task_id -> (gpus, actual_duration)

    get_critical_path_priority.cache = {}
    task_list_internal.sort(
        key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks:
                continue
            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue

            k = sp_map[task_key]
            base_duration = proc_times[task_key]
            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0
            task_id, stage = task_key

            producer_gpus, producer_duration = None, 0.0
            if stage == 'DIT' and task_id in vae_placements:
                producer_gpus, producer_duration = vae_placements[task_id]

            gpus, start, end, penalty = find_best_gpu_allocation_v7_joint(
                gpu_timeline, k, base_duration,
                earliest_start_time=dep_finish_time,
                topology_model=topology_model,
                producer_gpus=producer_gpus,
                producer_duration=producer_duration,
                lookahead_threshold=3, beam_width=5,
                task_key=task_key, log_file=log_file,
            )
            if gpus is None:
                continue

            for g in gpus:
                gpu_timeline[g] = end
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage, k, end - start, start, end, dataset_id, gpus, penalty))
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            if stage == 'VAE':
                vae_placements[task_id] = (gpus, end - start)
            scheduled_this_loop = True
            break

        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("Scheduling deadlock: check resources and dependencies.")

    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


def evaluate_chromosome_wrapper_joint(chromosome, tasks_input, total_gpus, topology_model):
    """Per-chromosome evaluation for the joint variant (parallel workers)."""
    _, makespan = generate_trace_from_chromosome_v3_joint(
        tasks_input, chromosome, total_gpus, topology_model)
    return makespan


def genetic_algorithm_schedule_v3_joint(
    tasks_input: List[Dict],
    total_gpus: int,
    population_size=50,
    num_generations=50,
    mutation_rate=0.1,
    elitism_size=2,
    log_file_path: str = None,
    machine_size: int = None,
    progress_recorder: list = None,
):
    """FULL mapper (joint intra+inter; Table 3 'full'): identical GA
    skeleton to v2_origin, only evaluation/final trace use the joint
    objective, keeping all ablation variants comparable."""
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler (Joint Intra+Inter) ---")

    if machine_size is None:
        machine_size = total_gpus
    try:
        topology_model = TopologyModel(params_file='fit_results.json', machine_size=machine_size, enable_topology=True)
    except Exception as e:
        print(f"❌ Error initializing TopologyModel: {e}")
        print("Falling back to topology-agnostic scheduling (all penalties = 1.0).")
        topology_model = type('DummyModel', (), {'predict_penalty': lambda self, gl: 1.0,
                                                 'machine_size': machine_size})()
    log_file = open(log_file_path, 'w') if log_file_path else None

    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')
    env_cores = os.environ.get("GA_NUM_CORES")
    if env_cores is not None:
        num_cores_to_use = min(int(env_cores), population_size)
    else:
        num_cores_to_use = min(os.cpu_count() or 64, population_size)
    print(f"🚀 Activating Parallel Evaluation on {num_cores_to_use} cores!")

    import time
    ga_start_time = time.time()
    if progress_recorder is not None:
        progress_recorder.clear()
    for gen in tqdm(range(num_generations), desc="Evolving Generations (Joint)"):
        makespan_results = Parallel(n_jobs=num_cores_to_use,
                                    backend='loky')(
                    delayed(evaluate_chromosome_wrapper_joint)(
                        chrom, tasks_input, total_gpus, topology_model
                    ) for chrom in population
                )
        fitnesses = {i: ms for i, ms in enumerate(makespan_results)}
        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            if progress_recorder is not None:
                progress_recorder.append({
                    "generation": gen + 1,
                    "best_makespan": best_overall_fitness,
                    "elapsed_plan_time": time.time() - ga_start_time,
                })
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")

        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        while len(new_population) < population_size:
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

            def mutate(chromosome):
                mutated_chromosome = list(chromosome)
                for i in range(len(mutated_chromosome)):
                    if random.random() < mutation_rate:
                        mutated_chromosome[i] = random.choice(sp_options_per_task[i])
                return mutated_chromosome

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        population = new_population

    print(f"\n[GA-Joint] Evolution complete. Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA-Joint] Best SP configuration (chromosome): {best_overall_chromosome}")

    # Final trace also uses the joint objective (v2_origin evaluates and
    # finalizes with different generators; fixed here).
    final_trace, final_makespan = generate_trace_from_chromosome_v3_joint(
        tasks_input, best_overall_chromosome, total_gpus, topology_model)
    final_trace_formatted = [item[:-1] for item in final_trace]
    return final_trace_formatted, final_makespan


def append_schedule_summary(
    *,
    schedule_type: str,
    n_gpus: int,
    machine_size: int,
    iteration: int,
    makespan: float,
    plan_time: float,
    yaml_path: str,
    output_dir: str = "schedule_summary",
):
    """Append one iteration's scheduling summary as a JSONL line."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{schedule_type}_ngpu{n_gpus}.jsonl"
    path = os.path.join(output_dir, filename)
    import json
    from datetime import datetime
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "schedule_type": schedule_type,
        "n_gpus": n_gpus,
        "machine_size": machine_size,
        "makespan": makespan,
        "plan_time_sec": plan_time,
        "yaml_path": yaml_path,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def schedule_random(num_runs=100, schedule_type="a_star"):
    total_gpus = 8
    makespans = []
    durations = []

    for i in range(num_runs):
        dataset_exec_times = generate_fake_dataset_exec_times(num_datasets=4)

        global DATASET_EXEC_TIMES
        DATASET_EXEC_TIMES = dataset_exec_times

        print(DATASET_EXEC_TIMES)

        tasks = simulate_tasks_variable(num_tasks=4)

        start = time.time()
        if schedule_type == "a_star":
            result = a_star_schedule(tasks, total_gpus=total_gpus)
            makespans.append(result.total_time)
        else:
            result = greedy_schedule(tasks, total_gpus=total_gpus)
            trace, makespan = greedy_schedule(tasks, total_gpus=8)
            makespans.append(makespan)
        end = time.time()

        if result:
            print(f"run {i}: {end - start:.2f}s")

            durations.append(end - start)
        else:
            print(f"run {i}: no feasible schedule")
            makespans.append(None)
            durations.append(end - start)

    valid_makespans = [m for m in makespans if m is not None]
    print("\nStatistics (successful runs):")
    print(f"total runs: {num_runs}, successful: {len(valid_makespans)}")
    print(f"max makespan: {max(valid_makespans):.2f} s")


def append_iteration_pareto(
    output_dir: str,
    schedule_type: str,
    n_gpus: int,
    machine_size: int,
    iteration: int,
    pareto_process: list,
    yaml_path: str,
):
    """Append one iteration's anytime-Pareto record as a JSONL line."""
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir,
        f"pareto_{schedule_type}_{n_gpus}gpus.jsonl"
    )

    record = {
        "schedule_type": schedule_type,
        "n_gpus": n_gpus,
        "machine_size": machine_size,
        "iteration": iteration,
        "yaml_path": yaml_path,
        "pareto": pareto_process,
        "num_points": len(pareto_process),
        "final_makespan": pareto_process[-1]["best_makespan"]
            if pareto_process else None,
    }
    import json
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def schedule(yaml_path: str, type="a_star", tasks: List[Dict] = None, 
             n_gpus=8, machine_size: int = None, use_topology: bool = True):

    if tasks is None:
        tasks = simulate_tasks_variable(num_tasks=4)
    else:
        print("Using provided tasks for scheduling.")

    now = time.time()

    result = None

    
    if type == "a_star":
        result = a_star_schedule(tasks, total_gpus=n_gpus)
    elif type == "greedy":
        result = greedy_schedule(tasks, total_gpus=n_gpus)
    elif type == "genetic":
        # Intra-only topology-aware GA (the production scheduler for the main
        # experiments). Ablation variants share the same call signature:
        #   genetic_algorithm_schedule_v1      -> no mapper   (wo_mapper)
        #   genetic_algorithm_schedule_v1_5    -> inter-only  (wo_intra)
        #   genetic_algorithm_schedule_v2      -> intra-only, use_topology flag
        #   type == "genetic_joint" below      -> FULL mapper (intra+inter)
        process = []
        trace, makespan = genetic_algorithm_schedule_v2_origin(
            tasks,
            total_gpus=n_gpus,
            population_size=200,
            num_generations=100,
            log_file_path=None,
            machine_size=machine_size,
            progress_recorder=process,
        # log_file_path="t2v_flow/planner/generated_schedules/wan/result.log"
        )
        
        Result = namedtuple('Result', ['trace', 'total_time'])
        result = Result(trace=trace, total_time=makespan)
    elif type == "genetic_joint":
        # FULL mapper: joint intra (topology-penalty beam placement) +
        # inter (cascade data affinity). Ablation map: genetic(v2_origin) =
        # wo_inter, v1_5 = wo_intra, v1/resource_only = wo_mapper.
        process = []
        trace, makespan = genetic_algorithm_schedule_v3_joint(
            tasks,
            total_gpus=n_gpus,
            population_size=200,
            num_generations=100,
            log_file_path=None,
            machine_size=machine_size,
            progress_recorder=process,
        )
        Result = namedtuple('Result', ['trace', 'total_time'])
        result = Result(trace=trace, total_time=makespan)
    elif type == "flex_sp":
        print(f"🚀 Starting FlexSP Scheduling for {len(tasks)} tasks...")
        from .flexsp_scheduler import flex_sp_schedule
        trace, makespan = flex_sp_schedule(tasks, total_gpus=n_gpus)
        if trace:
            Result = namedtuple('Result', ['trace', 'total_time'])
            result = Result(trace=trace, total_time=makespan) 
        else:
            print("❌ FlexSP Scheduling failed (likely OOM).")

    elif type == "megatron-lm":
        print(f"🚀 Starting Naive Megatron-LM Scheduling for {len(tasks)} tasks...")
        from .megatron_scheduler import megatron_schedule
        trace, makespan = megatron_schedule(tasks, total_gpus=n_gpus)
        if trace:
            Result = namedtuple('Result', ['trace', 'total_time'])
            result = Result(trace=trace, total_time=makespan)

    elif type == "heft_cpsat":
        print(f"🚀 Starting HEFT + CP-SAT Scheduling for {len(tasks)} tasks...")
        from .heft_cpsat_scheduler import heft_cpsat_schedule
        cpsat_time_limit = float(os.environ.get("CPSAT_TIME_LIMIT", "60"))
        trace, makespan = heft_cpsat_schedule(
            tasks,
            total_gpus=n_gpus,
            machine_size=machine_size,
            use_topology=use_topology,
            cpsat_time_limit=cpsat_time_limit,
        )
        if trace:
            Result = namedtuple('Result', ['trace', 'total_time'])
            result = Result(trace=trace, total_time=makespan)

    elif type == "heft":
        print(f"🚀 Starting HEFT Scheduling for {len(tasks)} tasks...")
        from .heft_cpsat_scheduler import heft_schedule
        if machine_size is None:
            machine_size = n_gpus
        try:
            topology_model = TopologyModel(
                params_file='fit_results.json',
                machine_size=machine_size,
                enable_topology=use_topology,
            )
        except Exception:
            topology_model = None
        trace, makespan = heft_schedule(
            tasks, total_gpus=n_gpus,
            topology_model=topology_model,
            use_topology=use_topology,
        )
        if trace:
            Result = namedtuple('Result', ['trace', 'total_time'])
            result = Result(trace=trace, total_time=makespan)

    if result and result.trace:
        trace_csv_path = yaml_path.replace(".yaml", "_trace.csv")

        df = pd.DataFrame(
            result.trace,
            columns=[
                "Task_ID",
                "Stage",
                "GPUs_Count",
                "Time",
                "Start",
                "End",
                "DatasetID",
                "GPU_List",
            ],
        )

        df["Makespan"] = result.total_time

        df.to_csv(trace_csv_path, index=False)
        print(f"Detailed schedule trace saved to {trace_csv_path}")

        print(df)
        print(f"\nBest makespan = {result.total_time:.2f}s")
        print(f"Scheduling took {time.time() - now:.2f}s")
        plan_time = time.time() - now

        # ===============================
        # ===============================
        try:
            iteration = None
            basename = os.path.basename(yaml_path)
            if basename.startswith("schedule_"):
                iteration = int(basename.replace("schedule_", "").replace(".yaml", ""))

            append_schedule_summary(
                schedule_type=type,
                n_gpus=n_gpus,
                machine_size=machine_size,
                iteration=iteration,
                makespan=result.total_time,
                plan_time=plan_time,
                yaml_path=yaml_path,
                output_dir=os.path.join(os.path.dirname(yaml_path), "summary"),
            )

            if type == "genetic" and process:
                try:
                    pareto_dir = os.path.join(
                        os.path.dirname(yaml_path),
                        "pareto"
                    )

                    append_iteration_pareto(
                        output_dir=pareto_dir,
                        schedule_type=type,
                        n_gpus=n_gpus,
                        machine_size=machine_size,
                        iteration=iteration,
                        pareto_process=process,
                        yaml_path=yaml_path,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to record Pareto process: {e}")

        except Exception as e:
            print(f"⚠️ Failed to record schedule summary: {e}")
        save_yaml_plan_from_trace(result.trace, yaml_path)

        plot_schedule_by_gpu(
            result.trace,
            total_gpus=n_gpus,
            save_path=yaml_path.replace(".yaml", ".png"),
        )

    else:
        print("No feasible schedule found")


def trace_to_yaml_with_correct_deps(trace: List[Tuple]) -> List[Dict]:
    """Convert a trace into the executor-compatible YAML task list
    (uppercase task_type; DIT tasks carry data_source_task)."""
    task_info_map = {}
    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in trace:
        task_key = (task_id, stage)
        gpu_str = "".join(str(g) for g in sorted(gpu_list))
        full_name = f"{dataset_id}_{stage.upper()}_g{gpu_str}"

        task_info_map[task_key] = {
            "full_name": full_name,
            "gpus": sorted(gpu_list),
            "sp": sp,
            "start_time": start,
            "stage": stage
        }

    sorted_trace_keys = sorted(task_info_map.keys(), key=lambda k: task_info_map[k]["start_time"])

    final_tasks_dict = {}
    last_task_on_gpu = {}

    for task_key in sorted_trace_keys:
        task_id, stage = task_key
        current_info = task_info_map[task_key]
        current_task_name = current_info["full_name"]

        dependencies = set()
        for gpu in current_info["gpus"]:
            if gpu in last_task_on_gpu:
                dependencies.add(last_task_on_gpu[gpu])

        if stage.upper() == "DIT":
            vae_key = (task_id, 'VAE')
            if vae_key in task_info_map:
                dependencies.add(task_info_map[vae_key]["full_name"])

        task_dict = {
            "name": current_task_name,
            "gpus": current_info["gpus"],
            "dependencies": sorted(list(dependencies)),
            "args": {
                "task_type": stage.upper(),
                "sp": current_info["sp"],
            },
        }
        
        if stage.upper() == "DIT":
            vae_key = (task_id, 'VAE')
            if vae_key in task_info_map:
                vae_info = task_info_map[vae_key]
                task_dict["args"]["data_source_task"] = vae_info["full_name"]


        final_tasks_dict[current_task_name] = task_dict

        for gpu in current_info["gpus"]:
            last_task_on_gpu[gpu] = current_task_name

    final_task_list = []
    added_tasks_names = set()
    for task_id, stage, _, _, _, _, _, _ in trace:
        task_key = (task_id, stage)
        if task_key in task_info_map:
            task_name = task_info_map[task_key]["full_name"]
            if task_name not in added_tasks_names:
                final_task_list.append(final_tasks_dict[task_name])
                added_tasks_names.add(task_name)

    return final_task_list


def find_conflict_task_names(trace: List[Tuple]) -> set:
    """Return the set of task names whose GPU intervals overlap."""
    conflicted_names = set()

    for i in range(len(trace)):
        id_i, _, _, _, start_i, end_i, dataset_i, gpus_i = trace[i]
        gpu_str_i = "".join(str(g) for g in sorted(gpus_i))
        name_i = f"{dataset_i}_{trace[i][1]}_g{gpu_str_i}"

        for j in range(i + 1, len(trace)):
            id_j, _, _, _, start_j, end_j, dataset_j, gpus_j = trace[j]
            gpu_str_j = "".join(str(g) for g in sorted(gpus_j))
            name_j = f"{dataset_j}_{trace[j][1]}_g{gpu_str_j}"

            if end_i > start_j and end_j > start_i:
                overlap = set(gpus_i) & set(gpus_j)
                if overlap:
                    conflicted_names.add(name_i)
                    conflicted_names.add(name_j)

    return conflicted_names


def save_yaml_plan_from_trace(
    trace: List[Tuple], yaml_path: str = "generated_plan.yaml"
):
    print(f"Trying to transform to yaml")
    tasks_yaml = trace_to_yaml_with_correct_deps(trace)

    tasks_yaml = {"tasks": tasks_yaml}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, yaml_path)

    print(f"Trying to save to: {save_path}")
    with open(save_path, "w") as f:
        yaml.dump(tasks_yaml, f, sort_keys=False)
    print(f"Schedule plan saved to {save_path}")


def yaml_to_trace(yaml_path: str) -> Tuple[List[Tuple], Dict[str, List[str]]]:
    """Parse a YAML plan back into a trace list of
    (task_id, stage, sp, duration=0, start=0, end=0, dataset_id, gpu_list)."""
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
        yaml_tasks = content["tasks"]

    trace = []
    dependency_map = {}

    for task in yaml_tasks:
        name = task["name"]
        stage = task["args"]["task_type"]
        sp = task["args"]["sp"]
        gpus = task["gpus"]
        dataset_id = "_".join(name.split("_")[:2])  # D_001
        deps = task.get("dependencies", [])

        trace.append((name, stage, sp, 0.0, 0.0, 0.0, dataset_id, gpus))
        dependency_map[name] = deps

    return trace, dependency_map


def topological_sort_trace(trace: List[Tuple], dependency_map: dict) -> List[Tuple]:
    name_to_trace = {t[0]: t for t in trace}
    visited = set()
    sorted_trace = []

    while len(visited) < len(trace):
        progress = False
        for name, deps in dependency_map.items():
            if name in visited:
                continue
            if all(d in visited for d in deps):
                sorted_trace.append(name_to_trace[name])
                visited.add(name)
                progress = True
        if not progress:
            raise RuntimeError("Topological sort failed: cyclic dependency detected")
    return sorted_trace


def replay_schedule(
    trace: List[Tuple],
    dependency_map: dict,
    dataset_exec_times: dict,
    total_gpus: int = 8,
) -> Tuple[List[Tuple], float]:
    gpu_timeline = [0.0] * total_gpus
    stage_end_time = {}
    finished = set()
    replayed_trace = []

    for task_id, stage, sp, _, _, _, dataset_id, gpu_list in trace:
        for dep in dependency_map.get(task_id, []):
            if dep not in finished:
                raise RuntimeError(f"dependency {dep} of {task_id} not finished")

        dataset = next(
            (v for v in dataset_exec_times.values() if v["name"] == dataset_id), None
        )
        if not dataset:
            raise KeyError(f"dataset not found: {dataset_id}")
        duration = dataset[stage.lower()][sp]

        ready_time = max(gpu_timeline[g] for g in gpu_list)
        for dep in dependency_map.get(task_id, []):
            ready_time = max(ready_time, stage_end_time[dep])

        start = ready_time
        end = start + duration
        for g in gpu_list:
            gpu_timeline[g] = end

        stage_end_time[task_id] = end
        finished.add(task_id)

        replayed_trace.append(
            (task_id, stage, sp, duration, start, end, dataset_id, gpu_list)
        )

    makespan = max(gpu_timeline)
    print(f"✅ Replay Makespan: {makespan:.2f}s")
    return replayed_trace, makespan


if __name__ == "__main__":
    schedule("generated_plan.yaml")


# if __name__ == "__main__":


