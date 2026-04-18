import json
import pandas as pd
import time
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from itertools import combinations

# ==============================================================================
# 1. 通信模型与辅助函数 (无改动)
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

def get_node_id(gpu_id: int) -> int:
    return gpu_id // CommunicationModelConfig.GPUS_PER_NODE

def calculate_intra_task_penalty(gpu_list: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTRA_TASK_PENALTY: return 0.0, "Intra-task penalty disabled."
    k = len(gpu_list)
    if k <= 1 or k > CommunicationModelConfig.INTRA_TASK_PENALTY_MAX_SP: return 0.0, "No penalty applied."
    node_counts = {}
    for gpu in gpu_list:
        node = get_node_id(gpu)
        node_counts[node] = node_counts.get(node, 0) + 1
    if len(node_counts) <= 1: return 0.0, "All GPUs on a single node."
    counts = list(node_counts.values())
    base_penalty = CommunicationModelConfig.INTRA_TASK_BASE_PENALTY * (len(node_counts) - 1)
    imbalance_penalty = np.std(counts) * CommunicationModelConfig.INTRA_TASK_IMBALANCE_FACTOR
    total_penalty = base_penalty + imbalance_penalty
    return total_penalty, f"Cross-node penalty: {node_counts}"

def calculate_inter_task_communication_cost(vae_gpus: List[int], dit_gpus: List[int]) -> Tuple[float, str]:
    if not CommunicationModelConfig.ENABLE_INTER_TASK_COST or not vae_gpus or not dit_gpus: return 0.0, "N/A"
    vae_gpu_set, dit_gpu_set = set(vae_gpus), set(dit_gpus)
    if vae_gpu_set == dit_gpu_set: return 0.0, "Full Overlap"
    if vae_gpu_set.intersection(dit_gpu_set): return CommunicationModelConfig.COST_BROADCAST, "Partial Overlap (Broadcast)"
    return CommunicationModelConfig.COST_P2P_BROADCAST, "No Overlap (P2P+Broadcast)"

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    if not hasattr(get_critical_path_priority, "cache"): get_critical_path_priority.cache = {}
    if task_key in get_critical_path_priority.cache: return get_critical_path_priority.cache[task_key]
    successors = [t for t, preds in predecessors.items() if task_key in preds]
    if not successors: return proc_times.get(task_key, float('inf'))
    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times) for succ in successors)
    result = proc_times.get(task_key, float('inf')) + max_succ_path
    get_critical_path_priority.cache[task_key] = result
    return result

# ==============================================================================
# 2. 通信感知的GPU决策函数 (无改动)
# ==============================================================================

def find_best_communication_aware_gpus(
    task_key: Tuple[int, str], k: int, gpu_timeline: List[float], proc_time: float,
    predecessor_info: Dict[str, Any]
) -> Dict[str, Any]:
    total_gpus = len(gpu_timeline)
    candidate_gpu_sets = []
    num_nodes = (total_gpus + CommunicationModelConfig.GPUS_PER_NODE - 1) // CommunicationModelConfig.GPUS_PER_NODE
    for i in range(num_nodes):
        node_gpus_indices = [g for g in range(i * CommunicationModelConfig.GPUS_PER_NODE, (i + 1) * CommunicationModelConfig.GPUS_PER_NODE) if g < total_gpus]
        if len(node_gpus_indices) >= k:
            sorted_node_gpus = sorted(node_gpus_indices, key=lambda g: gpu_timeline[g])
            candidate_gpu_sets.append(tuple(sorted(sorted_node_gpus[:k])))
    if task_key[1] == 'DIT' and predecessor_info.get('gpus'):
        vae_gpus = predecessor_info['gpus']
        if len(vae_gpus) == k: candidate_gpu_sets.append(tuple(sorted(vae_gpus)))
        elif len(vae_gpus) < k:
            remaining_needed = k - len(vae_gpus)
            other_gpus = sorted([g for g in range(total_gpus) if g not in vae_gpus], key=lambda g: gpu_timeline[g])
            if len(other_gpus) >= remaining_needed: candidate_gpu_sets.append(tuple(sorted(vae_gpus + other_gpus[:remaining_needed])))
    sorted_global_gpus = sorted(range(total_gpus), key=lambda g: gpu_timeline[g])
    candidate_gpu_sets.append(tuple(sorted(sorted_global_gpus[:k])))
    pool_size = min(total_gpus, k + 4)
    if pool_size > k:
        for combo in combinations(sorted_global_gpus[:pool_size], k): candidate_gpu_sets.append(tuple(sorted(combo)))
    unique_candidates = sorted(list(set(candidate_gpu_sets)))
    if not unique_candidates: return None
    best_assignment = {'finish_time': float('inf')}
    for gpus_tuple in unique_candidates:
        gpus = list(gpus_tuple)
        intra_penalty, _ = calculate_intra_task_penalty(gpus)
        inter_cost = 0.0
        if task_key[1] == 'DIT' and predecessor_info.get('gpus'):
            inter_cost, _ = calculate_inter_task_communication_cost(predecessor_info['gpus'], gpus)
        gpus_ready_time = max(gpu_timeline[g] for g in gpus)
        dependency_ready_time = predecessor_info.get('end_time', 0.0) + inter_cost
        start_time = max(gpus_ready_time, dependency_ready_time)
        adjusted_duration = proc_time + intra_penalty
        finish_time = start_time + adjusted_duration
        if finish_time < best_assignment['finish_time']:
            best_assignment = {
                'gpus': gpus, 'start_time': start_time, 'end_time': finish_time,
                'finish_time': finish_time, 'duration': adjusted_duration,
                'inter_cost': inter_cost, 'intra_penalty': intra_penalty
            }
    return best_assignment if best_assignment['finish_time'] != float('inf') else None

# ==============================================================================
# 3. 核心调度函数 (增加健壮性检查)
# ==============================================================================

def generate_trace_from_chromosome(tasks_input: List[Dict], chromosome: List[int], total_gpus: int, 
                                     priority_strategy="critical_path") -> Tuple[List[Tuple], float]:
    # a. 数据预处理
    task_list_internal, proc_times, sp_map, predecessors = [], {}, {}, {}
    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae, key_dit = (task_id, 'VAE'), (task_id, 'DIT')
        sp_vae_options, sp_dit_options = dict(task_def['vae']), dict(task_def['dit'])
        sp_vae, sp_dit = chromosome[chromosome_idx], chromosome[chromosome_idx + 1]
        task_list_internal.extend([key_vae, key_dit])
        proc_times[key_vae] = sp_vae_options.get(sp_vae, float('inf'))
        sp_map[key_vae] = sp_vae
        proc_times[key_dit] = sp_dit_options.get(sp_dit, float('inf'))
        sp_map[key_dit] = sp_dit
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2

    # b. 预计算CP值
    get_critical_path_priority.cache = {}
    task_priorities = {t: get_critical_path_priority(t, predecessors, proc_times) for t in task_list_internal}
    
    # c. 调度逻辑
    gpu_timeline, trace, stage_end_time = [0.0] * total_gpus, [], {}
    completed_tasks, task_gpu_assignments = set(), {}

    if priority_strategy in ["critical_path", "max_concurrency"]:
        task_list_internal.sort(key=lambda t: task_priorities[t] if priority_strategy == "critical_path" else (len(predecessors.get(t, [])), proc_times[t]), reverse=(priority_strategy == "critical_path"))
        ready_queue = list(task_list_internal)
        while len(completed_tasks) < len(proc_times):
            scheduled_this_loop = False
            for task_key in list(ready_queue):
                if not all(dep in completed_tasks for dep in predecessors.get(task_key, [])):
                    continue
                
                k = sp_map[task_key]
                base_proc_time = proc_times[task_key]
                
                # ==================== 新增：健壮性检查 ====================
                if base_proc_time == float('inf'):
                    raise ValueError(f"任务 {task_key} 的SP值({k})无效, 导致执行时间无穷大。\n"
                                     f"请检查遗传算法的染色体生成/变异逻辑是否正确。")
                # ========================================================
                
                pred_info = {}
                if task_preds := predecessors.get(task_key, []):
                    pred_key = task_preds[0]
                    pred_info = {'end_time': stage_end_time.get(pred_key, 0.0), 'gpus': task_gpu_assignments.get(pred_key)}
                
                best_fit = find_best_communication_aware_gpus(task_key, k, gpu_timeline, base_proc_time, pred_info)
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
                dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
                trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id, gpus, inter_comm_cost, intra_task_penalty))
                break

            if not scheduled_this_loop and ready_queue:
                raise RuntimeError("调度死锁！没有可调度的任务，但队列不为空。")
    else:
        raise NotImplementedError(f"策略 '{priority_strategy}' 尚未实现。")

    return trace, (max(gpu_timeline) if gpu_timeline else 0.0)

# ==============================================================================
# 4. 顶层遗传算法 (已修正Bug)
# ==============================================================================

def genetic_algorithm_schedule(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2,
                               priority_strategy="critical_path"):
    print(f"\n--- 🚀 Starting Communication-Aware Genetic Algorithm (Strat: {priority_strategy}) ---")
    
    # ==================== Bug修正区域 START ====================
    # 修正sp_options_per_task的构建逻辑，确保与染色体解析顺序一致
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])
    # ==================== Bug修正区域 END ======================

    population = [[random.choice(options) for options in sp_options_per_task] for _ in range(population_size)]
    best_overall_chromosome, best_overall_fitness = None, float('inf')

    for gen in tqdm(range(num_generations), desc=f"Evolving (Strat: {priority_strategy})"):
        fitnesses = {}
        for i, chrom in enumerate(population):
            _, makespan = generate_trace_from_chromosome(tasks_input, chrom, total_gpus, priority_strategy)
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
            parent1 = random.choice(sorted_population[:int(population_size * 0.2)])
            parent2 = random.choice(sorted_population[:int(population_size * 0.2)])
            point = random.randint(1, len(parent1) - 1)
            child = parent1[:point] + parent2[point:]
            
            # ==================== Bug修正区域 START ====================
            # 修正变异逻辑，使用正确的选项列表
            mutated_child = list(child)
            for i in range(len(mutated_child)):
                if random.random() < mutation_rate:
                    mutated_child[i] = random.choice(sp_options_per_task[i])
            # ==================== Bug修正区域 END ======================
            new_population.append(mutated_child)
        population = new_population

    print(f"\n[GA] Evolution complete. Best makespan: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration: {best_overall_chromosome}")
    final_trace, final_makespan = generate_trace_from_chromosome(tasks_input, best_overall_chromosome, total_gpus, priority_strategy)
    
    return final_trace, final_makespan

# ==============================================================================
# 5. 分析与主评估流程 (无改动)
# ==============================================================================

def evaluate_schedule_from_file(tasks_json_path: str, n_gpus: int, priority_strategy: str):
    try:
        with open(tasks_json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"✅ Loaded {len(tasks)} tasks from '{tasks_json_path}'")
    except Exception as e:
        print(f"❌ Error loading tasks: {e}"); return

    if not tasks: print("⚠️ No tasks found."); return

    start_time = time.time()
    final_trace, final_makespan = genetic_algorithm_schedule(
        tasks_input=tasks, total_gpus=n_gpus,
        population_size=10, num_generations=200,
        mutation_rate=0.1, elitism_size=2,
        priority_strategy=priority_strategy
    )
    scheduling_duration = time.time() - start_time
    print(f"⏱️  Scheduling finished in {scheduling_duration:.2f}s")

    if not final_trace: print("❌ GA did not return a valid trace."); return

    trace_df = pd.DataFrame(final_trace,
        columns=["Task_ID", "Stage", "GPUs_Count", "Time_w_Penalty", "Start", "End", 
                 "DatasetID", "GPU_List", "Inter_Cost_s", "Intra_Penalty_s"]
    ).sort_values(by=['Start', 'End']).reset_index(drop=True)

    print("\n--- Final Communication-Aware Schedule Analysis ---")
    print(trace_df.to_string())

    total_inter_comm = trace_df['Inter_Cost_s'].sum()
    total_intra_penalty = trace_df['Intra_Penalty_s'].sum()
    total_comm_overhead = total_inter_comm + total_intra_penalty
    total_execution_time = trace_df['Time_w_Penalty'].sum()
    base_proc_time_sum = total_execution_time - total_intra_penalty
    overall_comm_percentage = (total_comm_overhead / total_execution_time * 100) if total_execution_time > 0 else 0

    print("\n--- Overall Statistics ---")
    print(f"Final Makespan: {final_makespan:.2f} s")
    print(f"Total task execution times (incl. penalty): {total_execution_time:.2f} s")
    print(f"  - Sum of base processing times: {base_proc_time_sum:.2f} s")
    print(f"  - Sum of intra-task penalties (asymmetry): {total_intra_penalty:.2f} s")
    print(f"Sum of inter-task communication costs (VAE->DIT): {total_inter_comm:.2f} s")
    print(f"Total communication overhead: {total_comm_overhead:.2f} s")
    print(f"Overall Communication Overhead Percentage: {overall_comm_percentage:.2f}%")
    
    output_csv_path = tasks_json_path.replace(".json", f"_analysis_comm_aware_{priority_strategy}.csv")
    trace_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Detailed analysis saved to '{output_csv_path}'")

if __name__ == '__main__':
    TASKS_FILE_PATH = "generated_schedules/wan/schedule_4_tasks.json"
    TOTAL_GPUS = 16
    PRIORITY_STRATEGY = "critical_path"
    
    print("\n" + "="*80)
    print(" C O M M U N I C A T I O N - A W A R E   S C H E D U L E R ")
    print("="*80 + "\n")
    print("--- Communication Model Configuration ---")
    print(f"  - GPUs per Node: {CommunicationModelConfig.GPUS_PER_NODE}")
    print(f"  - Intra-Task Penalty: {'Enabled' if CommunicationModelConfig.ENABLE_INTRA_TASK_PENALTY else 'Disabled'}")
    print(f"  - Inter-Task Cost: {'Enabled' if CommunicationModelConfig.ENABLE_INTER_TASK_COST else 'Disabled'}")
    
    evaluate_schedule_from_file(TASKS_FILE_PATH, TOTAL_GPUS, PRIORITY_STRATEGY)