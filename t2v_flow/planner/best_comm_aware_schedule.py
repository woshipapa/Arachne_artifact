import json
import pandas as pd
import time
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from itertools import combinations

# ==============================================================================
# 1. 通信模型配置
# ==============================================================================
class CommunicationModelConfig:
    GPUS_PER_NODE = 8
    ENABLE_INTRA_TASK_PENALTY = True
    INTRA_TASK_PENALTY_MAX_SP = 8
    INTRA_TASK_BASE_PENALTY = 2.5 
    INTRA_TASK_IMBALANCE_FACTOR = 1.5
    ENABLE_INTER_TASK_COST = True
    COST_BROADCAST = 3.0
    COST_P2P_BROADCAST = 5.0

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def get_node_id(gpu_id: int) -> int:
    return gpu_id // CommunicationModelConfig.GPUS_PER_NODE

def calculate_intra_task_penalty(gpu_list: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTRA_TASK_PENALTY: return 0.0, "Disabled"
    k = len(gpu_list)
    if k <= 1 or k > CommunicationModelConfig.INTRA_TASK_PENALTY_MAX_SP: return 0.0, "No penalty"
    node_counts = {}
    for gpu in gpu_list:
        node = get_node_id(gpu)
        node_counts[node] = node_counts.get(node, 0) + 1
    if len(node_counts) <= 1: return 0.0, "All GPUs on single node"
    counts = list(node_counts.values())
    base_penalty = CommunicationModelConfig.INTRA_TASK_BASE_PENALTY * (len(node_counts) - 1)
    imbalance_penalty = np.std(counts) * CommunicationModelConfig.INTRA_TASK_IMBALANCE_FACTOR
    total_penalty = base_penalty + imbalance_penalty
    return total_penalty, f"Cross-node penalty: {node_counts}"

def calculate_inter_task_communication_cost(vae_gpus: List[int], dit_gpus: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTER_TASK_COST or not vae_gpus or not dit_gpus: return 0.0, "N/A"
    vae_gpu_set, dit_gpu_set = set(vae_gpus), set(dit_gpus)
    if vae_gpu_set == dit_gpu_set: return 0.0, "Full Overlap"
    if vae_gpu_set.intersection(dit_gpu_set): return CommunicationModelConfig.COST_BROADCAST, "Partial Overlap"
    return CommunicationModelConfig.COST_P2P_BROADCAST, "No Overlap"

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    if not hasattr(get_critical_path_priority, "cache"): get_critical_path_priority.cache = {}
    if task_key in get_critical_path_priority.cache: return get_critical_path_priority.cache[task_key]
    successors = [t for t, preds in predecessors.items() if task_key in preds]
    if not successors: return proc_times.get(task_key, float('inf'))
    max_succ_path = max(get_critical_path_priority(s, predecessors, proc_times) for s in successors)
    result = proc_times.get(task_key, float('inf')) + max_succ_path
    get_critical_path_priority.cache[task_key] = result
    return result

# ==============================================================================
# 3. 新版 GPU 分配函数（融合量化模型和 NIC 亲和）
# ==============================================================================
def find_best_communication_aware_gpus(
    task_key: Tuple[int, str], k: int, gpu_timeline: List[float], proc_time: float,
    predecessor_info: Dict[str, Any], nic_affinity: Dict[int, List[int]],
    nic_bandwidth: Dict[int, float], pattern_time_lookup: Dict[Tuple[int,int], float],
    max_wait_time: float = 10.0
) -> Dict[str, Any]:
    alpha, rho, sigma = 1.0, 0.5, 0.2
    total_gpus = len(gpu_timeline)
    NICs = list(nic_bandwidth.keys())
    candidate_gpu_sets = []

    # 单节点优先
    num_nodes = (total_gpus + CommunicationModelConfig.GPUS_PER_NODE - 1) // CommunicationModelConfig.GPUS_PER_NODE
    for node_id in range(num_nodes):
        node_gpus = [g for g in range(node_id*CommunicationModelConfig.GPUS_PER_NODE,
                                     min((node_id+1)*CommunicationModelConfig.GPUS_PER_NODE, total_gpus))]
        if len(node_gpus) >= k:
            candidate_gpu_sets.append(node_gpus[:k])

    # 跨节点 top-M 子集
    top_M_per_node = 3
    for node_id in range(num_nodes):
        node_gpus = [g for g in range(node_id*CommunicationModelConfig.GPUS_PER_NODE,
                                     min((node_id+1)*CommunicationModelConfig.GPUS_PER_NODE, total_gpus))]
        if len(node_gpus) > 0:
            selected, covered_nics = [], set()
            for g in sorted(node_gpus, key=lambda g: gpu_timeline[g]):
                new_nics = set(nic_affinity[g]) - covered_nics
                if new_nics:
                    selected.append(g)
                    covered_nics.update(new_nics)
                if len(selected) == k: break
            if selected: candidate_gpu_sets.append(selected)

    # 全局 top-k
    sorted_gpus = sorted(range(total_gpus), key=lambda g: gpu_timeline[g])
    candidate_gpu_sets.append(sorted_gpus[:k])
    candidate_gpu_sets = [list(x) for x in set(tuple(sorted(c)) for c in candidate_gpu_sets)]
    if not candidate_gpu_sets: return None

    best_assignment = {'score': -float('inf')}
    current_time = min(gpu_timeline)

    for gpus in candidate_gpu_sets:
        intra_penalty, _ = calculate_intra_task_penalty(gpus)
        inter_cost = 0.0
        if task_key[1] == 'DIT' and predecessor_info.get('gpus'):
            inter_cost, _ = calculate_inter_task_communication_cost(predecessor_info['gpus'], gpus)

        ready_gpu_time = max(gpu_timeline[g] for g in gpus)
        dep_ready_time = predecessor_info.get('end_time', 0.0) + inter_cost
        start_time = max(ready_gpu_time, dep_ready_time)

        # 使用测量数据预测 T_i
        pattern_type = len(set(get_node_id(g) for g in gpus))
        measured_proc_time = pattern_time_lookup.get((k, pattern_type), proc_time)
        adjusted_duration = measured_proc_time + intra_penalty

        # wait / frag
        wait_time = max(0.0, start_time - current_time)
        wait_penalty = wait_time / max_wait_time if wait_time > 0 else 0.0
        all_nics = [nic_affinity[g] for g in gpus]
        flat_nics = [n for sub in all_nics for n in sub]
        frag_penalty = np.std([flat_nics.count(n) for n in set(flat_nics)])  

        # score
        score = -adjusted_duration - alpha*wait_penalty - rho*frag_penalty + sigma*len(set(flat_nics))

        if score > best_assignment['score']:
            best_assignment = {
                'gpus': gpus,
                'start_time': start_time,
                'end_time': start_time + adjusted_duration,
                'duration': adjusted_duration,
                'inter_cost': inter_cost,
                'intra_penalty': intra_penalty,
                'wait_penalty': wait_penalty,
                'frag_penalty': frag_penalty,
                'score': score
            }
    return best_assignment if best_assignment['score'] > -float('inf') else None

# ==============================================================================
# 4. GA 调度核心流程
# ==============================================================================
def generate_trace_from_chromosome(tasks_input: List[Dict], chromosome: List[int],
                                    total_gpus: int, nic_affinity: Dict[int,List[int]],
                                    nic_bandwidth: Dict[int,float],
                                    pattern_time_lookup: Dict[Tuple[int,int], float],
                                    priority_strategy="critical_path") -> Tuple[List[Tuple], float]:

    task_list_internal, proc_times, sp_map, predecessors = [], {}, {}, {}
    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')
        sp_vae_options, sp_dit_options = dict(task_def['vae']), dict(task_def['dit'])
        sp_vae, sp_dit = chromosome[chromosome_idx], chromosome[chromosome_idx+1]
        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = sp_vae_options.get(sp_vae, float('inf'))
        sp_map[key_vae] = sp_vae
        proc_times[key_dit] = sp_dit_options.get(sp_dit, float('inf'))
        sp_map[key_dit] = sp_dit
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2

    get_critical_path_priority.cache = {}
    task_priorities = {t: get_critical_path_priority(t, predecessors, proc_times) for t in task_list_internal}

    gpu_timeline, trace, stage_end_time = [0.0]*total_gpus, [], {}
    completed_tasks, task_gpu_assignments = set(), {}

    if priority_strategy in ["critical_path", "max_concurrency"]:
        task_list_internal.sort(key=lambda t: task_priorities[t] if priority_strategy=="critical_path"
                                else (len(predecessors.get(t,[])), proc_times[t]), reverse=True)
        ready_queue = list(task_list_internal)
        while len(completed_tasks) < len(proc_times):
            scheduled_this_loop = False
            for task_key in list(ready_queue):
                if not all(dep in completed_tasks for dep in predecessors.get(task_key, [])): continue
                k = sp_map[task_key]
                base_proc_time = proc_times[task_key]
                if base_proc_time == float('inf'):
                    raise ValueError(f"Invalid SP {k} for {task_key}")

                pred_info = {}
                if task_preds := predecessors.get(task_key, []):
                    pred_key = task_preds[0]
                    pred_info = {'end_time': stage_end_time.get(pred_key,0.0),
                                 'gpus': task_gpu_assignments.get(pred_key)}

                best_fit = find_best_communication_aware_gpus(
                    task_key, k, gpu_timeline, base_proc_time, pred_info,
                    nic_affinity, nic_bandwidth, pattern_time_lookup
                )
                if best_fit is None: continue

                gpus, start, end, duration = best_fit['gpus'], best_fit['start_time'], best_fit['end_time'], best_fit['duration']
                inter_comm_cost, intra_task_penalty = best_fit['inter_cost'], best_fit['intra_penalty']
                for g in gpus: gpu_timeline[g] = end
                stage_end_time[task_key] = end
                completed_tasks.add(task_key)
                task_gpu_assignments[task_key] = gpus
                ready_queue.remove(task_key)
                scheduled_this_loop = True

                task_id, stage = task_key
                dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id']==task_id)
                trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id,
                              gpus, inter_comm_cost, intra_task_penalty))
                break

            if not scheduled_this_loop and ready_queue:
                raise RuntimeError("Deadlock: no schedulable tasks left")
    else:
        raise NotImplementedError(f"Strategy {priority_strategy} not implemented")

    return trace, max(gpu_timeline) if gpu_timeline else 0.0

# ==============================================================================
# 5. GA 调度
# ==============================================================================
def genetic_algorithm_schedule(tasks_input: List[Dict], total_gpus: int,
                               nic_affinity: Dict[int,List[int]], nic_bandwidth: Dict[int,float],
                               pattern_time_lookup: Dict[Tuple[int,int], float],
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2,
                               priority_strategy="critical_path"):

    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]
    best_overall_chromosome, best_overall_fitness = None, float('inf')

    for gen in tqdm(range(num_generations), desc="GA Evolution"):
        fitnesses = {}
        for i, chrom in enumerate(population):
            _, makespan = generate_trace_from_chromosome(
                tasks_input, chrom, total_gpus,
                nic_affinity, nic_bandwidth, pattern_time_lookup, priority_strategy
            )
            fitnesses[i] = makespan

        min_fitness_idx = min(fitnesses, key=fitnesses.get)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            tqdm.write(f"Gen {gen+1}: New best makespan = {best_overall_fitness:.2f}s")

        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])

        while len(new_population) < population_size:
            parent1 = random.choice(sorted_population[:int(population_size*0.2)])
            parent2 = random.choice(sorted_population[:int(population_size*0.2)])
            point = random.randint(1, len(parent1)-1)
            child = parent1[:point] + parent2[point:]
            mutated_child = list(child)
            for i in range(len(mutated_child)):
                if random.random() < mutation_rate:
                    mutated_child[i] = random.choice(sp_options_per_task[i])
            new_population.append(mutated_child)
        population = new_population

    final_trace, final_makespan = generate_trace_from_chromosome(
        tasks_input, best_overall_chromosome, total_gpus,
        nic_affinity, nic_bandwidth, pattern_time_lookup, priority_strategy
    )
    return final_trace, final_makespan, best_overall_chromosome

