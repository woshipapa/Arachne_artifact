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
    """
    自动生成符合 DATASET_EXEC_TIMES 格式的伪数据。
    时间满足：SP 数越大，时间越短（模拟并行加速）。
    VAE 时间范围：[25, 120]，DIT 时间范围：[25, 241]
    """
    SP_STRATEGIES = [1, 2, 4, 8]
    fake_exec_times = {}

    for i in range(num_datasets):
        name = f"D_{i:03d}"

        # 先生成 base 时间（SP=1）
        base_vae = random.uniform(20, 60)  # 起点在 80~120
        base_dit = random.uniform(40, 100)  # 起点在 120~241

        # 每增加一倍卡数，执行时间减少一定比例
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
                "dataset_id": name,  # ✅ 加上 dataset 编号
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
    """辅助函数，用于生成所有可能的跨机GPU分配方案 (Split Plans)。"""
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
    # 资源×时间 优先
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
    """
    【V1 改进版GPU选择器】
    具备Inter-Cascade Awareness的GPU选择函数。
    
    Args:
        gpu_timeline: 当前所有GPU的预计空闲时间。
        k: 需要选择的GPU数量。
        preferred_gpus: (可选) 上一个阶段（VAE）使用的GPU列表。
    
    Returns:
        一个元组，包含选择的GPU列表和它们的就绪时间。
    """
    
    # 如果没有提供偏好，或者偏好列表为空，则行为与原始的 find_earliest_gpus 完全相同。
    if not preferred_gpus:
        # 创建一个 (空闲时间, GPU索引) 的元组列表
        all_gpus_by_time = [(gpu_timeline[i], i) for i in range(len(gpu_timeline))]
        # 使用 heapq 高效找到时间最早的 k 个
        earliest_k_gpus = heapq.nsmallest(k, all_gpus_by_time)
        
        selected = [gpu_idx for time, gpu_idx in earliest_k_gpus]
        # 就绪时间是这k个中最后一个变为空闲的时间
        start_time = max(time for time, gpu_idx in earliest_k_gpus) if earliest_k_gpus else 0.0
        return selected, start_time

    # --- 如果提供了偏好的GPU列表 ---
    
    # 1. 将GPU分为“偏好组”和“其他组”，并按空闲时间排序
    preferred_candidates = []
    other_candidates = []
    
    preferred_set = set(preferred_gpus)
    for gpu_idx, free_time in enumerate(gpu_timeline):
        candidate = (free_time, gpu_idx)
        if gpu_idx in preferred_set:
            preferred_candidates.append(candidate)
        else:
            other_candidates.append(candidate)
            
    # 对两个组分别进行排序，确保组内是时间最优的
    preferred_candidates.sort()
    other_candidates.sort()
    
    # 2. 优先从“偏好组”中选择，如果不够再从“其他组”中补充
    selected_gpus = []
    if len(preferred_candidates) >= k:
        # 偏好组的GPU足够，直接从里面选最早的k个
        selected_gpus = preferred_candidates[:k]
    else:
        # 偏好组的GPU不够，全部选中，再从其他组里补充
        selected_gpus.extend(preferred_candidates)
        needed = k - len(preferred_candidates)
        selected_gpus.extend(other_candidates[:needed])
        
    # 3. 提取最终结果并计算就绪时间
    final_gpu_list = [gpu_idx for time, gpu_idx in selected_gpus]
    final_start_time = max(time for time, gpu_idx in selected_gpus) if selected_gpus else 0.0
    
    return final_gpu_list, final_start_time


def _get_smart_local_candidates(gpus_on_machine: List[int], k: int, gpu_timeline: List[float], topology_model: TopologyModel) -> List[Tuple]:
    """【新增】辅助函数：智能地从单台机器的可用GPU中，生成一小批高质量候选组合"""
    if len(gpus_on_machine) < k: return []
    
    candidates = set()
    
    # 策略1: 最早可用的k个
    candidates.add(tuple(sorted(sorted(gpus_on_machine, key=lambda g: gpu_timeline[g])[:k])))
    
    # 策略2: 按GPU ID排序（物理位置）
    sorted_by_id = sorted(gpus_on_machine)
    candidates.add(tuple(sorted_by_id[:k])) # ID最低的k个
    if len(sorted_by_id) > k: candidates.add(tuple(sorted_by_id[-k:])) # ID最高的k个

    # 策略3: NUMA感知选择
    numa0_gpus = [g for g in gpus_on_machine if topology_model._gpu_to_group_label(g) in ['A', 'B']]
    numa1_gpus = [g for g in gpus_on_machine if topology_model._gpu_to_group_label(g) in ['C', 'D']]
    if len(numa0_gpus) >= k: candidates.add(tuple(sorted(numa0_gpus)[:k]))
    if len(numa1_gpus) >= k: candidates.add(tuple(sorted(numa1_gpus)[:k]))
        
    return list(candidates)

# 根据惩罚和 NUMA 距离找出一批候选集，目标是makespan最小, 这个方案候选集的组合过于复杂

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

        # --- 新增: 记录详细评分到日志 ---
        if log_file:
            log_file.write(f"--- Decision for Task {task_key} at t_start={t_start:.2f} ---\n")
            log_file.write(f"Found {len(scored_candidates)} candidates. Evaluating top {beam_width} (Beam Width).\n")
            log_file.write("Rank | Penalty  | NUMA Dist | GPU List               | Status\n")
            log_file.write("---- | -------- | --------- | ---------------------- | ------\n")
            for i, (penalty, distance, gpu_list) in enumerate(scored_candidates):
                status = "IN BEAM" if i < beam_width else "Pruned"
                # 为了日志整洁，只记录前20名和在beam中的
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

    # ---- 限制 search_times（非常重要）----
    search_times = sorted(t for t in set(gpu_timeline) if t >= earliest_start_time)
    search_times = [earliest_start_time] + search_times[:6]  # 👈 截断

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

        # ---- machine 分组 ----
        machine_groups = defaultdict(list)
        for g in available:
            machine_groups[g // topology_model.machine_size].append(g)

        # ---- cheap score ----
        def cheap_score(gpu_tuple):
            nodes = {g // topology_model.machine_size for g in gpu_tuple}
            span = gpu_tuple[-1] - gpu_tuple[0]
            return (len(nodes), span)

        # ---- 生成 split plans ----
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
            # 🔥 Beam merge（替代 itertools.product）
            # =====================================================
            merged_candidates = beam_merge_candidates(
                per_machine_candidates,
                topology_model,
                beam_width=beam_width * 4,
                cheap_score_fn=cheap_score,
            )

            # =====================================================
            # Expensive topology scoring（已合并）
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


# cheap filting
# def find_best_gpu_allocation_v6(
#     gpu_timeline: List[float],
#     k: int,
#     base_duration: float,
#     earliest_start_time: float,
#     topology_model: TopologyModel,
#     lookahead_threshold: float = 20.0,
#     beam_width: int = 5,
#     task_key: Tuple = None,
#     log_file: IO = None
# ) -> Tuple[List[int], float, float, float]:

#     if k == 0:
#         return [], earliest_start_time, earliest_start_time, 1.0

#     potential_start_times = sorted(set(gpu_timeline))
#     search_times = sorted(
#         t for t in potential_start_times if t >= earliest_start_time
#     )
#     search_times = [earliest_start_time] + search_times

#     best = {
#         "gpus": None,
#         "start": float('inf'),
#         "finish": float('inf'),
#         "penalty": float('inf'),
#     }

#     for t_start in search_times:
#         if best["gpus"] and t_start > best["start"] + lookahead_threshold:
#             break

#         available = [i for i, t in enumerate(gpu_timeline) if t <= t_start]
#         if len(available) < k:
#             continue

#         # =========================================================
#         # Phase 0: 构建候选集（与你原来一致）
#         # =========================================================
#         smart_candidates = set()
#         machine_groups = defaultdict(list)

#         for g in available:
#             machine_groups[g // topology_model.machine_size].append(g)

#         for gpus_on_machine in machine_groups.values():
#             smart_candidates.update(
#                 _get_smart_local_candidates(
#                     gpus_on_machine, k, gpu_timeline, topology_model
#                 )
#             )

#         split_plans = _generate_split_plans(k, machine_groups)

#         for plan in split_plans:
#             num_nodes = len(plan)
#             limit = 3 if num_nodes >= 4 else 5 if num_nodes == 3 else 10

#             per_machine = []
#             valid = True
#             for m_id, cnt in plan.items():
#                 cands = _get_smart_local_candidates(
#                     machine_groups[m_id], cnt, gpu_timeline, topology_model
#                 )
#                 if not cands:
#                     valid = False
#                     break
#                 per_machine.append(cands[:limit])

#             if not valid:
#                 continue

#             for parts in itertools.product(*per_machine):
#                 combo = tuple(sorted(g for p in parts for g in p))
#                 smart_candidates.add(combo)

#         if not smart_candidates:
#             continue

#         # =========================================================
#         # Phase 1: Cheap filtering（🔥 核心加速点）
#         # =========================================================
#         def cheap_score(gpu_tuple):
#             nodes = {g // topology_model.machine_size for g in gpu_tuple}
#             span = gpu_tuple[-1] - gpu_tuple[0]
#             return (len(nodes), span)

#         smart_candidates = list(smart_candidates)
#         smart_candidates.sort(key=cheap_score)

#         PRE_N = min(len(smart_candidates), beam_width * 20)
#         cheap_top = smart_candidates[:PRE_N]

#         # =========================================================
#         # Phase 2: Expensive topology scoring（只算少量）
#         # =========================================================
#         scored = []
#         for gpu_tuple in cheap_top:
#             penalty = topology_model.predict_penalty(gpu_tuple)
#             dist = topology_model.calculate_numa_distance(gpu_tuple)
#             scored.append((penalty, dist, list(gpu_tuple)))

#         scored.sort(key=lambda x: (x[0], x[1]))
#         topk = scored[:beam_width]

#         for penalty, _, gpus in topk:
#             finish = t_start + base_duration * penalty
#             if finish < best["finish"]:
#                 best.update(
#                     gpus=gpus,
#                     start=t_start,
#                     finish=finish,
#                     penalty=penalty,
#                 )

#     if best["gpus"] is None:
#         idx = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])[:k]
#         start = max(gpu_timeline[i] for i in idx)
#         penalty = topology_model.predict_penalty(tuple(idx))
#         finish = start + base_duration * penalty
#         return idx, start, finish, penalty

#     return best["gpus"], best["start"], best["finish"], best["penalty"]




def beam_merge_candidates(
    per_machine_candidates,
    topology_model,
    beam_width,
    cheap_score_fn,
):
    """
    per_machine_candidates: List[List[Tuple[int]]]
      - 每个元素是一台机器的候选 GPU tuples（已排序/截断）
    返回：List[Tuple[int]]，size ≤ beam_width
    """
    beam = [()]  # partial gpu tuples

    for machine_cands in per_machine_candidates:
        new_beam = []

        for partial in beam:
            for cand in machine_cands:
                merged = tuple(sorted(partial + cand))
                new_beam.append(merged)

        # ---- cheap prune ----
        new_beam.sort(key=cheap_score_fn)
        new_beam = new_beam[:beam_width * 2]  # 预留一点余量

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
    # 找最早的 k 个可用 GPU slot
    available_times = sorted(gpu_timeline)
    
    # 第 k 个最早可用时间
    start = max(earliest_start_time, available_times[k - 1])
    finish = start + base_duration

    # 随便选 k 个 GPU id（前 k 个即可）
    gpus = list(range(k))
    return gpus, start, finish, 1.0


# 最终版
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

# --- 核心优化：使用智能启发式生成小而精的候选集 ---
        smart_candidates, machine_groups = set(), defaultdict(list)
        for gpu in available_gpus: machine_groups[gpu // topology_model.machine_size].append(gpu)
        
        # 1. 生成高质量的“节点内”候选
        for _, gpus_on_machine in machine_groups.items():
            smart_candidates.update(_get_smart_local_candidates(gpus_on_machine, k, gpu_timeline, topology_model))
        
        # 2. 生成高质量的“跨节点”候选
        split_plans = _generate_split_plans(k, machine_groups)

        
        for plan in split_plans:

            # --- START: MODIFICATION 1 ---
            # 动态控制：根据计划跨越的节点数，动态设定每个节点贡献的候选数量上限
            # 目标是让总组合数保持在 100 左右，与之前2节点时的情况类似
            num_nodes_in_plan = len(plan)
            if num_nodes_in_plan >= 4:
                # print("enter 4 nodes=======")
                limit = 3  # 4节点时, 3^4 = 81 组合
            elif num_nodes_in_plan == 3:
                limit = 5  # 3节点时, 5^3 = 125 组合
            else:  # 2个或更少节点
                limit = 10 # 2节点时, 10^2 = 100 组合
            # --- END: MODIFICATION 1 ---
            # 对每个分配方案，从涉及的每台机器上获取一小批精英局部候选
            machine_local_candidates = []
            valid_plan = True
            for machine_id, count in plan.items():
                local_candidates = _get_smart_local_candidates(machine_groups[machine_id], count, gpu_timeline, topology_model)
                if not local_candidates:
                    valid_plan = False
                    break
                machine_local_candidates.append(local_candidates[:limit])
            if not valid_plan: continue

            # 将多台机器的精英局部候选进行笛卡尔积
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

        # --- 新增: 记录详细评分到日志 ---
        if log_file:
            log_file.write(f"--- Decision for Task {task_key} at t_start={t_start:.2f} ---\n")
            log_file.write(f"Found {len(scored_candidates)} candidates. Evaluating top {beam_width} (Beam Width).\n")
            log_file.write("Rank | Penalty  | NUMA Dist | GPU List               | Status\n")
            log_file.write("---- | -------- | --------- | ---------------------- | ------\n")
            for i, (penalty, distance, gpu_list) in enumerate(scored_candidates):
                status = "IN BEAM" if i < beam_width else "Pruned"
                # 为了日志整洁，只记录前20名和在beam中的
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
    """
    Buddy system GPU 分配：在 GPU timeline 中查找 k=2^n 个连续且按对齐块分配的 GPU。
    :param gpu_timeline: 每个 GPU 当前的可用时间（浮点数）
    :param k: 需要的 GPU 数量（必须为 2 的幂次）
    :return: 选中的 GPU 列表，以及其最早开始时间
    """
    assert (k & (k - 1)) == 0, "k 必须是 2 的幂次"

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

        # --- VAE 阶段 ---
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
            raise RuntimeError(f"VAE阶段无法为任务 {i} 分配 {k} 个 GPU")

        # --- DIT 阶段 ---
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
            raise RuntimeError(f"DIT阶段无法为任务 {i} 分配 {k} 个 GPU")

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

            # VAE 阶段未完成
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

            # VAE 完成但 DIT 未完成
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
    """
    根据调度 trace 生成任务时间线图。
    :param trace: 调度轨迹列表，元素为 (task_id, stage, gpus, duration, start, end, dataset_id)
    :param save_path: 若提供路径则保存图片，否则展示
    """
    df = pd.DataFrame(
        trace, columns=["Task", "Stage", "GPUs", "Time", "Start", "End", "DatasetID"]
    )

    # ✅ 添加任务标签 Task N [Ddataset_id]
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
            label=row["Stage"] if idx < 2 else "",  # 避免重复图例
        )
        # 添加任务信息注释
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
        print(f"📷 图像已保存至: {save_path}")
    else:
        plt.show()


def plot_schedule_by_gpu(trace, total_gpus=8, save_path=None):
    """
    根据调度 trace 按 GPU rank 显示任务时间线（每个 GPU 一行）
    :param trace: 调度轨迹列表，元素为 (task_id, stage, gpus, duration, start, end, dataset_id, gpu_list)
    :param total_gpus: GPU 数量
    :param save_path: 若提供路径则保存图片，否则展示
    """
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
        print(f"📷 图像已保存至: {save_path}")
    else:
        plt.show()



# =============================================================================
# 核心修改 1: 插入完整的遗传算法 (GA) 调度器
# =============================================================================

# --- 2.1 GA 的辅助函数：关键路径和列表调度器 ---

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    """计算任务的关键路径长度（从当前任务到最终完成），用于任务优先级排序。"""
    if not hasattr(get_critical_path_priority, "cache"):
        get_critical_path_priority.cache = {}
    if task_key in get_critical_path_priority.cache:
        return get_critical_path_priority.cache[task_key]

    # 找到所有直接依赖于当前任务的后续任务
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
            raise RuntimeError("调度死锁！")

    return trace, max(gpu_timeline) if gpu_timeline else 0.0



# =============================================================================
# @profile
def generate_trace_from_chromosome_v2_origin(tasks_input: List[Dict], chromosome: List[int], total_gpus: int, 
                                     topology_model: TopologyModel, log_file: IO = None) -> Tuple[List[Tuple], float]: # 添加 topology_model 参数
    """
    【桥梁函数 - V2 拓扑感知版】
    接收GA找到的最优SP组合(chromosome)，并使用高效的拓扑感知列表调度算法生成最终的trace。
    """
    # a. 数据预处理 (与之前相同)
    task_list_internal = []
    proc_times = {} # 这里存储的是“基准”处理时间
    sp_map = {}
    predecessors = {}
    
    # ... (这部分数据填充逻辑和您原来的一样，无需改动)
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

    # b. 拓扑感知列表调度器核心逻辑
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    # 任务优先级排序 (与之前相同)
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks: continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue
            
            # --- 核心修改点 ---
            k = sp_map[task_key]
            base_duration = proc_times[task_key]
            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0

            # 调用全新的、拓扑感知的分配函数
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
            
            if gpus is None: # 理论上新函数总能找到方案，除非k>total_gpus
                continue

            # 使用新函数返回的结果更新所有状态
            actual_duration = end - start
            for g in gpus:
                gpu_timeline[g] = end
            
            task_id, stage = task_key
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            # 在trace中可以记录更多信息，如惩罚系数
            trace.append((task_id, stage, k, actual_duration, start, end, dataset_id, gpus, penalty))
            
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            task_list_internal.remove(task_key)
            scheduled_this_loop = True
            break # 每次调度一个任务后，重新从高优先级任务开始检查
        
        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("调度死锁！请检查资源和依赖。")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan



def generate_trace_from_chromosome_v1_5_inter_aware(
    tasks_input: List[Dict], 
    chromosome: List[int], 
    total_gpus: int
) -> Tuple[List[Tuple], float]:
    """
    【桥梁函数 V1.5 - Inter-Cascade Aware】
    在V1的基础上，增加了对同一Sample的VAE和DIT任务的GPU亲和性调度。
    """
    # ... V1中相同的数据预处理部分 ...
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
        
    # b. 列表调度器核心逻辑
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    # ==================== 新增部分 ====================
    # 新增一个字典，用于存储每个VAE任务最终分配到的GPU
    vae_gpu_placements = {}
    # ===============================================

    # ... V1中相同的任务排序部分 ...
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal):
            if task_key in completed_tasks: continue

            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                continue
            
            k = sp_map[task_key]
            
            # ==================== 核心修改点 ====================
            task_id, stage = task_key
            preferred_gpus_for_dit = None
            
            # 1. 如果当前任务是DIT，查找其对应的VAE使用了哪些GPU
            if stage == 'DIT' and task_id in vae_gpu_placements:
                preferred_gpus_for_dit = vae_gpu_placements[task_id]

            # 2. 调用新的、具备级联感知的GPU选择器
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
            
            # ==================== 新增部分 ====================
            # 3. 如果刚刚调度的是VAE任务，记录它使用的GPU
            if stage == 'VAE':
                vae_gpu_placements[task_id] = gpus
            # ===============================================

            scheduled_this_loop = True
            break
        
        if not scheduled_this_loop and task_list_internal:
            raise RuntimeError("调度死锁！请检查资源和依赖。")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan

# --- 2.2 GA 进化逻辑 ---

def generate_trace_from_chromosome_v1(tasks_input: List[Dict], chromosome: List[int], total_gpus: int) -> Tuple[List[Tuple], float]:
    """
    【桥梁函数】
    接收GA找到的最优SP组合(chromosome)，并使用高效的列表调度算法生成最终的trace。
    这是连接GA和你的YAML生成器的关键。
    """
    # a. 数据预处理，以适配列表调度器
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}

    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae = (task_id, 'VAE')
        key_dit = (task_id, 'DIT')
        
        # VAE 任务信息
        sp_vae = chromosome[chromosome_idx]
        task_list_internal.append(key_vae)
        proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
        sp_map[key_vae] = sp_vae
        chromosome_idx += 1
        
        # DiT 任务信息
        sp_dit = chromosome[chromosome_idx]
        task_list_internal.append(key_dit)
        proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
        sp_map[key_dit] = sp_dit
        chromosome_idx += 1
        
        # 依赖关系
        predecessors[key_dit] = [key_vae]

    # b. 列表调度器核心逻辑
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    # 使用关键路径作为任务优先级
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

    while len(completed_tasks) < len(proc_times):
        scheduled_this_loop = False
        for task_key in list(task_list_internal): # 遍历副本
            if task_key in completed_tasks: continue

            # 检查所有前置依赖是否已完成
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
            raise RuntimeError("调度死锁！请检查资源和依赖。")
            
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
    local_k=5,          # 每代做 local search 的个体数
    local_tries=3,      # 每个个体 local 搜索次数
    progress_recorder: list = None,
):
    sp_options_per_gene = build_sp_options(tasks_input, total_gpus)

    # --- 预构建 eval 所需结构（避免重复 dict(...)）---
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
        # 生成每个 stage 的 k 与 duration
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

        # 记录 best
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

    # 生成最终 trace：用你原来的 generate_trace_from_chromosome_v1（但建议也换 heap/ready 改进版）
    final_trace, final_makespan = generate_trace_from_chromosome_v1(tasks_input, best_chrom, total_gpus)
    return final_trace, final_makespan


def genetic_algorithm_schedule_v1(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2, log_file_path: str = None):
    """
    这是新的顶层调度函数，它将取代 a_star_schedule 的决策功能。
    """
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    # a. 初始化，从你的tasks_input格式中提取GA需要的信息
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    # b. 创建初始种群(P 个染色体)
    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    # c. 进化循环
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        # 评估适应度 (这里只需要makespan, 所以可以用轻量级评估)
        fitnesses = {}


        # 这里对于种群中每个染色体进行评估
        for i, chrom in enumerate(population):
            # 为了快速评估，可以只计算makespan而不生成完整trace
            _, makespan = generate_trace_from_chromosome_v1(tasks_input, chrom, total_gpus)
            fitnesses[i] = makespan

        # 记录当代最佳
        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        # d. 创建下一代种群（选择、交叉、变异）
        new_population = []
        # 精英主义：直接保留最好的几个个体
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        # 锦标赛选择和繁衍
        while len(new_population) < population_size:
            # 选择
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)] # 倾向于选择好的
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            # 交叉
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            # 变异
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

    # e. 使用找到的最优染色体生成最终的、详细的trace
    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v1(tasks_input, best_overall_chromosome, total_gpus)
    
    return final_trace, final_makespan





def genetic_algorithm_schedule_v1_5(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2, log_file_path: str = None):
    """
    这是新的顶层调度函数，它将取代 a_star_schedule 的决策功能。
    """
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    # a. 初始化，从你的tasks_input格式中提取GA需要的信息
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    # b. 创建初始种群
    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    # c. 进化循环
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        # 评估适应度 (这里只需要makespan, 所以可以用轻量级评估)
        fitnesses = {}
        for i, chrom in enumerate(population):
            # 为了快速评估，可以只计算makespan而不生成完整trace
            _, makespan = generate_trace_from_chromosome_v1_5_inter_aware(tasks_input, chrom, total_gpus)
            fitnesses[i] = makespan

        # 记录当代最佳
        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            tqdm.write(f"Generation {gen+1}: New best makespan found: {best_overall_fitness:.2f}s")
            
        # d. 创建下一代种群（选择、交叉、变异）
        new_population = []
        # 精英主义：直接保留最好的几个个体
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        # 锦标赛选择和繁衍
        while len(new_population) < population_size:
            # 选择
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)] # 倾向于选择好的
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            # 交叉
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            # 变异
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

    # e. 使用找到的最优染色体生成最终的、详细的trace
    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v1_5_inter_aware(tasks_input, best_overall_chromosome, total_gpus)
    
    return final_trace, final_makespan



def evaluate_chromosome_wrapper(chromosome, tasks_input, total_gpus, topology_model, use_topology):
    """
    这是一个包装函数，用于将单个染色体的评估逻辑隔离出来。
    注意：这里的参数会被序列化(Pickle)发送到子进程。
    """
    # 这里调用你原来的评估函数
    _, makespan = generate_trace_from_chromosome_v2(
        tasks_input, 
        chromosome, 
        total_gpus, 
        topology_model=topology_model, 
        log_file=None, # 并行时通常不写日志，避免文件锁冲突
        topology_aware=use_topology
    )
    return makespan


def evaluate_chromosome_wrapper_origin(chromosome, tasks_input, total_gpus, topology_model):
    """
    这是一个包装函数，用于将单个染色体的评估逻辑隔离出来。
    注意：这里的参数会被序列化(Pickle)发送到子进程。
    """
    # 这里调用你原来的评估函数
    _, makespan = generate_trace_from_chromosome_v2_origin(
        tasks_input, 
        chromosome, 
        total_gpus, 
        topology_model=topology_model, 
        log_file=None, # 并行时通常不写日志，避免文件锁冲突
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
    """
    这是新的顶层调度函数，它将取代 a_star_schedule 的决策功能。
    """
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    if machine_size is None:
        machine_size = total_gpus
    try:
        topology_model = TopologyModel(params_file='fit_results.json', machine_size=machine_size, enable_topology=use_topology)
    except Exception as e:
        print(f"❌ Error initializing TopologyModel: {e}")
        print("调度将回退到拓扑无关模式（所有惩罚为1.0）。")
        # 创建一个“哑”模型作为备用
        topology_model = type('DummyModel', (), {'predict_penalty': lambda self, gl: 1.0})()
    log_file = open(log_file_path, 'w') if log_file_path else None
    # a. 初始化，从你的tasks_input格式中提取GA需要的信息
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    # b. 创建初始种群
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
    # c. 进化循环
    import time

    ga_start_time = time.time()

    if progress_recorder is not None:
        progress_recorder.clear()
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        # 评估适应度 (这里只需要makespan, 所以可以用轻量级评估)
        # fitnesses = {}
        # for i, chrom in enumerate(population):
        #     # 为了快速评估，可以只计算makespan而不生成完整trace
        #     _, makespan = generate_trace_from_chromosome_v2(tasks_input, chrom, total_gpus, topology_model=topology_model, log_file=log_file)
        #     fitnesses[i] = makespan
        makespan_results = Parallel(n_jobs=num_cores_to_use, 
                                    backend='loky')(
                    delayed(evaluate_chromosome_wrapper)(
                        chrom, tasks_input, total_gpus, topology_model, use_topology
                    ) for chrom in population
                )
        
        # 将结果转回字典格式，兼容后续逻辑
        fitnesses = {i: ms for i, ms in enumerate(makespan_results)}
        # 记录当代最佳
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
            
        # d. 创建下一代种群（选择、交叉、变异）
        new_population = []
        # 精英主义：直接保留最好的几个个体
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        # 锦标赛选择和繁衍
        while len(new_population) < population_size:
            # 选择
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)] # 倾向于选择好的
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            # 交叉
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            # 变异
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

    # e. 使用找到的最优染色体生成最终的、详细的trace
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
    """
    这是新的顶层调度函数，它将取代 a_star_schedule 的决策功能。
    """
    print("\n--- 🚀 Starting Genetic Algorithm Scheduler ---")
    
    if machine_size is None:
        machine_size = total_gpus
    try:
        topology_model = TopologyModel(params_file='fit_results.json', machine_size=machine_size, enable_topology=True)
    except Exception as e:
        print(f"❌ Error initializing TopologyModel: {e}")
        print("调度将回退到拓扑无关模式（所有惩罚为1.0）。")
        # 创建一个“哑”模型作为备用
        topology_model = type('DummyModel', (), {'predict_penalty': lambda self, gl: 1.0})()
    log_file = open(log_file_path, 'w') if log_file_path else None
    # a. 初始化，从你的tasks_input格式中提取GA需要的信息
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    # b. 创建初始种群
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
    # c. 进化循环
    import time

    ga_start_time = time.time()

    if progress_recorder is not None:
        progress_recorder.clear()
    for gen in tqdm(range(num_generations), desc="Evolving Generations"):
        # 评估适应度 (这里只需要makespan, 所以可以用轻量级评估)
        # fitnesses = {}
        # for i, chrom in enumerate(population):
        #     # 为了快速评估，可以只计算makespan而不生成完整trace
        #     _, makespan = generate_trace_from_chromosome_v2(tasks_input, chrom, total_gpus, topology_model=topology_model, log_file=log_file)
        #     fitnesses[i] = makespan
        makespan_results = Parallel(n_jobs=num_cores_to_use, 
                                    backend='loky')(
                    delayed(evaluate_chromosome_wrapper_origin)(
                        chrom, tasks_input, total_gpus, topology_model
                    ) for chrom in population
                )
        
        # 将结果转回字典格式，兼容后续逻辑
        fitnesses = {i: ms for i, ms in enumerate(makespan_results)}
        # 记录当代最佳
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
            
        # d. 创建下一代种群（选择、交叉、变异）
        new_population = []
        # 精英主义：直接保留最好的几个个体
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        # 锦标赛选择和繁衍
        while len(new_population) < population_size:
            # 选择
            parent1 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)] # 倾向于选择好的
            parent2 = sorted_population[min(range(len(sorted_population)), key=lambda x: random.random()**2)]
            # 交叉
            point = random.randint(1, len(parent1) - 1)
            child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            # 变异
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

    # e. 使用找到的最优染色体生成最终的、详细的trace
    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome_v2(tasks_input, best_overall_chromosome, total_gpus, topology_model=topology_model)
    
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
    """
    将每个 iteration 的调度摘要信息追加写入文件（JSONL）
    """
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

    # 追加写（JSONL，一行一个 iteration）
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def schedule_random(num_runs=100, schedule_type="a_star"):
    total_gpus = 8
    makespans = []
    durations = []

    for i in range(num_runs):
        # 生成符合顺序递减要求的伪数据集
        dataset_exec_times = generate_fake_dataset_exec_times(num_datasets=4)

        # 替换全局变量（或你可以传入参数）
        global DATASET_EXEC_TIMES
        DATASET_EXEC_TIMES = dataset_exec_times

        print(DATASET_EXEC_TIMES)

        # 模拟任务
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
            print(f"第 {i} 次模拟耗时：{end - start:.2f}s")

            durations.append(end - start)
        else:
            print(f"❌ 第 {i} 次模拟未找到调度解")
            makespans.append(None)
            durations.append(end - start)

    # 统计结果（剔除 None）
    valid_makespans = [m for m in makespans if m is not None]
    print("\n🔍 统计结果（基于成功调度）:")
    print(f"总次数: {num_runs}, 成功次数: {len(valid_makespans)}")
    print(f"最大 Makespan: {max(valid_makespans):.2f} s")
    # print(f"平均 Makespan: {np.mean(valid_makespans):.2f} s")
    # print(f"平均调度耗时: {np.mean(durations):.4f} s")



def append_iteration_pareto(
    output_dir: str,
    schedule_type: str,
    n_gpus: int,
    machine_size: int,
    iteration: int,
    pareto_process: list,
    yaml_path: str,
):
    """
    记录一个 iteration 的 Pareto / anytime 过程（一行）
    """
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
        "pareto": pareto_process,   # 👈 核心
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
        # 如果传入了 tasks，则不再模拟
        print("Using provided tasks for scheduling.")

    now = time.time()

    result = None # 先初始化 result

    
    if type == "a_star":
        result = a_star_schedule(tasks, total_gpus=n_gpus)
    elif type == "greedy":
        result = greedy_schedule(tasks, total_gpus=n_gpus)
    elif type == "genetic":
        # 贪心的寻找最早完成的k个GPU,无通信拓扑感知
        process = []
        # trace, makespan = genetic_algorithm_schedule_v1_improved(
        #     tasks_input=tasks,
        #     total_gpus=n_gpus,
        #     population_size=200,
        #     num_generations=400,
        #     mutation_rate=0.1,
        #     elitism_size=4,
        #     local_k=8,
        #     local_tries=3,
        #     progress_recorder=process,
        # )
        # trace, makespan = genetic_algorithm_schedule_v1(
        #     tasks_input=tasks,
        #     total_gpus=n_gpus,
        #     population_size=100,
        #     num_generations=200,
        #     log_file_path=None
        # )
        # greedy + inter-cascade
        # trace, makespan = genetic_algorithm_schedule_v1_5(tasks, total_gpus=n_gpus,
        #     population_size=100,
        #     num_generations=200,
        #     log_file_path=None,
        # )


        # 使用资源数量与拓扑感知的遗传算法调度 intra
        
        # trace, makespan = genetic_algorithm_schedule_v2(
        #     tasks,
        #     total_gpus=n_gpus,
        #     population_size=200,
        #     num_generations=100,
        #     log_file_path=None,
        #     machine_size=machine_size,
        #     progress_recorder=process,
        #     use_topology=use_topology
        # # log_file_path="t2v_flow/planner/generated_schedules/wan/result.log"
        # )

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
        
        # 为了完美兼容你后续的流程，我们创建一个与A*结果类似的轻量级对象
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

    if result and result.trace:  # 确保 result 和 trace 都有效
        # ⭐ --- 新增代码：保存详细 Trace 到 CSV 文件 ---
        # 1. 定义CSV文件名 (与YAML文件同名，扩展名不同)
        trace_csv_path = yaml_path.replace(".yaml", "_trace.csv")

        # 2. 将 trace 转换为 Pandas DataFrame
        df = pd.DataFrame(
            result.trace,
            columns=[
                "Task_ID",  # 改为 Task_ID 以示区分
                "Stage",
                "GPUs_Count",  # 改为 GPUs_Count
                "Time",
                "Start",
                "End",
                "DatasetID",
                "GPU_List",
            ],
        )

        # 3. 添加 Makespan 和其他元数据
        df["Makespan"] = result.total_time

        # 4. 保存到 CSV 文件
        df.to_csv(trace_csv_path, index=False)
        print(f"📄 已将详细调度轨迹保存到 {trace_csv_path}")
        # --- 新增代码结束 ---

        print(df)
        print(f"\n✅ 最优 Makespan = {result.total_time:.2f}s")
        print(f"⏱️ 调度耗时: {time.time() - now:.2f}s")
        plan_time = time.time() - now

        # ===============================
        # ⭐ 新增：记录 iteration 级别调度摘要
        # ===============================
        try:
            # 从 yaml_path 里解析 iteration（如果你是 schedule_XX.yaml）
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
        # 调用我们之前写好的函数来保存YAML
        save_yaml_plan_from_trace(result.trace, yaml_path)

        # 生成并保存GPU利用率图
        plot_schedule_by_gpu(
            result.trace,
            total_gpus=n_gpus,
            save_path=yaml_path.replace(".yaml", ".png"),
        )

    else:
        print("❌ 未找到调度方案")


def trace_to_yaml_with_correct_deps(trace: List[Tuple]) -> List[Dict]:
    """
    【最终版 - 兼容A*格式】
    根据trace生成与A*格式完全一致的YAML任务列表。
    - 确保 DiT 任务有 data_source_task
    - 移除 overlapped 和 data_source_rank 字段
    - 确保 task_type 为大写 (DIT, VAE)
    """
    # === 步骤 1: 预处理，将trace信息存入一个更易于访问的字典中 ===
    task_info_map = {}
    for task_id, stage, sp, t, start, end, dataset_id, gpu_list in trace:
        task_key = (task_id, stage)
        gpu_str = "".join(str(g) for g in sorted(gpu_list))
        # 你的命名格式: 1_129_1280_720_rank2_VAE_g0123...
        # dataset_id 就是那个包含 rank 的长字符串
        full_name = f"{dataset_id}_{stage.upper()}_g{gpu_str}"

        task_info_map[task_key] = {
            "full_name": full_name,
            "gpus": sorted(gpu_list),
            "sp": sp,
            "start_time": start,
            "stage": stage # 保留原始stage用于逻辑判断
        }

    # === 步骤 2: 按时间顺序构建包含完整依赖的任务字典 ===
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

        # === 核心修改区域: 构建与 A* 格式一致的字典 ===
        task_dict = {
            "name": current_task_name,
            "gpus": current_info["gpus"],
            "dependencies": sorted(list(dependencies)),
            "args": {
                # 1. 确保 task_type 是大写
                "task_type": stage.upper(),
                "sp": current_info["sp"],
            },
        }
        
        # 2. 如果是DIT任务，只添加 data_source_task
        if stage.upper() == "DIT":
            vae_key = (task_id, 'VAE')
            if vae_key in task_info_map:
                vae_info = task_info_map[vae_key]
                # 3. 添加必需的 data_source_task
                task_dict["args"]["data_source_task"] = vae_info["full_name"]
                # 4. 根据A*格式，【不】添加 data_source_rank

        # 5. 根据A*格式，【不】添加 overlapped 字段

        final_tasks_dict[current_task_name] = task_dict

        for gpu in current_info["gpus"]:
            last_task_on_gpu[gpu] = current_task_name

    # === 步骤 3: 按照原始 trace 的顺序组装最终的列表 ===
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
    """
    返回发生 GPU 冲突的任务名称集合（即会写入 YAML 的 task.name）
    """
    conflicted_names = set()

    for i in range(len(trace)):
        id_i, _, _, _, start_i, end_i, dataset_i, gpus_i = trace[i]
        gpu_str_i = "".join(str(g) for g in sorted(gpus_i))
        name_i = f"{dataset_i}_{trace[i][1]}_g{gpu_str_i}"  # stage 是 trace[i][1]

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

    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, yaml_path)

    print(f"Trying to save to: {save_path}")
    with open(save_path, "w") as f:
        yaml.dump(tasks_yaml, f, sort_keys=False)
    print(f"✅ 已将调度计划保存到 {save_path}")


def yaml_to_trace(yaml_path: str) -> Tuple[List[Tuple], Dict[str, List[str]]]:
    """
    从 YAML 文件中解析任务调度计划，恢复为 trace 列表。
    每项 trace 格式：
      (task_id, stage, sp, duration=0, start=0, end=0, dataset_id, gpu_list)
    """
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
        yaml_tasks = content["tasks"]

    trace = []
    dependency_map = {}

    for task in yaml_tasks:
        name = task["name"]  # 例: D_001_DIT_g0167
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
            raise RuntimeError("❌ 拓扑排序失败，可能存在循环依赖")
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
                raise RuntimeError(f"{task_id} 的依赖 {dep} 未完成")

        dataset = next(
            (v for v in dataset_exec_times.values() if v["name"] == dataset_id), None
        )
        if not dataset:
            raise KeyError(f"找不到数据集: {dataset_id}")
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


# 得到调度图和生成yml
if __name__ == "__main__":
    schedule("generated_plan.yaml")


# 根据yml重放调度图
# if __name__ == "__main__":
#     yaml_path = "/Users/xander/Documents/git/flex/Megatron_VAST/t2v_flow/planner/generated_plan.yaml"

#     # 1. 加载 trace 和依赖关系
#     trace, dependency_map = yaml_to_trace(yaml_path)

#     # 2. 拓扑排序（确保依赖顺序）
#     sorted_trace = topological_sort_trace(trace, dependency_map)

#     # 3. 执行调度重放
#     replayed_trace, makespan = replay_schedule(
#         sorted_trace,
#         dependency_map=dependency_map,
#         dataset_exec_times=DATASET_EXEC_TIMES,  # 可切换为 DATASET_EXEC_TIMES_20 或 120
#         total_gpus=8,
#     )

#     # 4. 可视化
#     plot_schedule_by_gpu(replayed_trace, total_gpus=8)
