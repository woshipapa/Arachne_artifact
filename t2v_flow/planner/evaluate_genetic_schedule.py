import json
import pandas as pd
import time
import random
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from itertools import combinations

# ==============================================================================
# ==============================================================================

def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    if k > len(gpu_timeline):
        return None, float('inf')
    sorted_gpu_indices = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
    selected_indices = sorted_gpu_indices[:k]
    start_time = max(gpu_timeline[i] for i in selected_indices)
    return selected_indices, start_time

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    if not hasattr(get_critical_path_priority, "cache"):
        get_critical_path_priority.cache = {}
    if task_key in get_critical_path_priority.cache:
        return get_critical_path_priority.cache[task_key]
    
    successors = [t for t, preds in predecessors.items() if task_key in preds]
    if not successors:
        return proc_times.get(task_key, float('inf'))
    
    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times) for succ in successors)
    result = proc_times.get(task_key, float('inf')) + max_succ_path
    get_critical_path_priority.cache[task_key] = result
    return result

# ==============================================================================
# ==============================================================================

def generate_trace_from_chromosome(tasks_input: List[Dict], chromosome: List[int], total_gpus: int, 
                                   priority_strategy="critical_path") -> Tuple[List[Tuple], float]:
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
        # ====================================================
        
        predecessors[key_dit] = [key_vae]
        chromosome_idx += 2

    get_critical_path_priority.cache = {}
    task_priorities = {t: get_critical_path_priority(t, predecessors, proc_times) for t in task_list_internal}
    
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()

    if priority_strategy in ["critical_path", "max_concurrency"]:
        if priority_strategy == "critical_path":
            task_list_internal.sort(key=lambda t: task_priorities[t], reverse=True)
        else: # max_concurrency
            task_list_internal.sort(key=lambda t: (len(predecessors.get(t, [])), proc_times[t]))

        ready_queue = list(task_list_internal)
        while len(completed_tasks) < len(proc_times):
            scheduled_this_loop = False
            for task_key in list(ready_queue):
                if not all(dep in completed_tasks for dep in predecessors.get(task_key, [])):
                    continue
                
                k = sp_map[task_key]
                gpus, start_time_gpu = find_earliest_gpus(gpu_timeline, k)
                
                if gpus is None: continue

                dep_finish_time = max([stage_end_time.get(dep, 0) for dep in predecessors.get(task_key, [])], default=0)
                start = max(start_time_gpu, dep_finish_time)
                duration = proc_times[task_key]
                end = start + duration

                for g in gpus: gpu_timeline[g] = end
                stage_end_time[task_key] = end
                completed_tasks.add(task_key)
                ready_queue.remove(task_key)
                scheduled_this_loop = True
                
                task_id, stage = task_key
                dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
                trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id, gpus))
                
                break

            if not scheduled_this_loop and ready_queue:
                raise RuntimeError("Scheduling deadlock!")

    elif priority_strategy == "exhaustive_combination":
        while len(completed_tasks) < len(task_list_internal):
            current_time = min(gpu_timeline) if gpu_timeline else 0.0
            free_gpu_indices = [i for i, t in enumerate(gpu_timeline) if t <= current_time]

            ready_tasks = [t for t in task_list_internal if t not in completed_tasks and all(d in completed_tasks for d in predecessors.get(t, []))]

            if not ready_tasks:
                if len(completed_tasks) < len(task_list_internal):
                    try:
                        next_event_time = min(t for t in gpu_timeline if t > current_time)
                        for i in range(len(gpu_timeline)):
                             if gpu_timeline[i] < next_event_time:
                                  gpu_timeline[i] = next_event_time
                        continue
                    except ValueError:
                        break 
                else:
                    break 

            best_combination, best_comb_priority_score = [], -1

            for r in range(len(ready_tasks), 0, -1):
                for combo in combinations(ready_tasks, r):
                    if sum(sp_map.get(t, float('inf')) for t in combo) <= len(free_gpu_indices):
                        combo_priority_score = sum(task_priorities.get(t, 0) for t in combo)
                        if combo_priority_score > best_comb_priority_score:
                            best_comb_priority_score = combo_priority_score
                            best_combination = list(combo)
            
            if not best_combination and ready_tasks:
                try:
                    next_event_time = min(t for t in gpu_timeline if t > current_time)
                    for i in range(len(gpu_timeline)):
                        if gpu_timeline[i] < next_event_time:
                             gpu_timeline[i] = next_event_time
                    continue
                except ValueError:
                    break

            temp_free_gpus = list(free_gpu_indices)
            for task_key in best_combination:
                k = sp_map[task_key]
                gpus = temp_free_gpus[:k]
                del temp_free_gpus[:k]
                
                start, duration = current_time, proc_times[task_key]
                end = start + duration

                for g in gpus: gpu_timeline[g] = end
                stage_end_time[task_key] = end
                completed_tasks.add(task_key)
                
                task_id, stage = task_key
                dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
                trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id, gpus))
    
    else:
        raise ValueError(f"Unknown priority strategy: {priority_strategy}")

    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


# ==============================================================================
# ==============================================================================

def genetic_algorithm_schedule(tasks_input: List[Dict], total_gpus: int,
                               population_size=50, num_generations=50, mutation_rate=0.1, elitism_size=2,
                               priority_strategy="critical_path"):
    print(f"\n--- 🚀 Starting Genetic Algorithm Scheduler (Strategy: {priority_strategy}) ---")
    
    sp_options_per_task = []
    for task_def in tasks_input:
        sp_options_per_task.append([opt[0] for opt in task_def['vae']])
        sp_options_per_task.append([opt[0] for opt in task_def['dit']])

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
            tqdm.write(f"Gen {gen+1}: New best makespan ({priority_strategy}) = {best_overall_fitness:.2f}s")
            
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses.values(), population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < population_size:
            parent1 = random.choice(sorted_population[:int(population_size * 0.2)])
            parent2 = random.choice(sorted_population[:int(population_size * 0.2)])
            point = random.randint(1, len(parent1) - 1)
            child = parent1[:point] + parent2[point:]
            mutated_child = list(child)
            for i in range(len(mutated_child)):
                if random.random() < mutation_rate:
                    mutated_child[i] = random.choice(sp_options_per_task[i])
            new_population.append(mutated_child)
        population = new_population

    print(f"\n[GA] Evolution complete ({priority_strategy}). Best makespan found: {best_overall_fitness:.2f}s")
    print(f"[GA] Best SP configuration: {best_overall_chromosome}")

    print("[GA] Generating final schedule trace with the best chromosome...")
    final_trace, final_makespan = generate_trace_from_chromosome(tasks_input, best_overall_chromosome, total_gpus, priority_strategy)
    
    return final_trace, final_makespan


# ==============================================================================
# ==============================================================================

def analyze_communication_cost(trace_df: pd.DataFrame, tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    comm_costs, comm_percentages = [], []
    tasks_lookup = {}
    for task in tasks:
        tasks_lookup[task['task_id']] = {
            "dit": {sp: time for sp, time in task['dit']},
            "vae": {sp: time for sp, time in task['vae']}
        }
    for _, row in trace_df.iterrows():
        task_id, stage, sp, total_time = row['Task_ID'], row['Stage'].lower(), row['GPUs_Count'], row['Time']
        try:
            baseline_time = tasks_lookup[task_id][stage][1]
        except KeyError:
            comm_costs.append(0); comm_percentages.append(0)
            continue
        if sp == 1: comm_cost = 0
        else: comm_cost = max(0, total_time - (baseline_time / sp))
        comm_costs.append(comm_cost)
        comm_percentages.append((comm_cost / total_time * 100) if total_time > 0 else 0)
    analyzed_df = trace_df.copy()
    analyzed_df['Communication_Cost_s'] = comm_costs
    analyzed_df['Communication_Percentage_%'] = comm_percentages
    return analyzed_df


def evaluate_schedule_from_file(tasks_json_path: str, n_gpus: int, priority_strategy: str):
    try:
        with open(tasks_json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"✅ Successfully loaded {len(tasks)} tasks from '{tasks_json_path}'")
    except Exception as e:
        print(f"❌ Error loading tasks file: {e}"); return

    if not tasks:
        print("⚠️ Warning: No tasks found in the file."); return

    start_time = time.time()
    final_trace, final_makespan = genetic_algorithm_schedule(
        tasks_input=tasks,
        total_gpus=n_gpus,
        population_size=200,
        num_generations=200,
        mutation_rate=0.1,
        elitism_size=2,
        priority_strategy=priority_strategy
    )
    scheduling_duration = time.time() - start_time
    print(f"⏱️ Scheduling finished in {scheduling_duration:.2f}s")

    if not final_trace:
        print("❌ Genetic algorithm did not return a valid trace."); return

    trace_df = pd.DataFrame(
        final_trace,
        columns=["Task_ID", "Stage", "GPUs_Count", "Time", "Start", "End", "DatasetID", "GPU_List"]
    ).sort_values(by=['Start', 'End']).reset_index(drop=True)

    print("\n🔬 Analyzing communication costs...")
    analyzed_df = analyze_communication_cost(trace_df, tasks)
    
    print("\n--- Scheduling and Communication Cost Analysis ---")
    print(analyzed_df.to_string())

    total_comm_cost = analyzed_df['Communication_Cost_s'].sum()
    total_execution_time = analyzed_df['Time'].sum()
    overall_comm_percentage = (total_comm_cost / total_execution_time * 100) if total_execution_time > 0 else 0

    print("\n--- Overall Statistics ---")
    print(f"Total Makespan: {final_makespan:.2f} s")
    print(f"Sum of all task execution times: {total_execution_time:.2f} s")
    print(f"Sum of all estimated communication costs: {total_comm_cost:.2f} s")
    print(f"Overall Communication Cost Percentage: {overall_comm_percentage:.2f}%")
    
    output_csv_path = tasks_json_path.replace(".json", f"_analysis_{priority_strategy}.csv")
    analyzed_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Detailed analysis saved to '{output_csv_path}'")


if __name__ == '__main__':
    # ===============================================================
    # ===============================================================
    TASKS_FILE_PATH = "generated_schedules/wan/schedule_4_tasks.json"
    TOTAL_GPUS = 16

    # ===============================================================
    # ===============================================================
    PRIORITY_STRATEGY = "critical_path"
    
    evaluate_schedule_from_file(TASKS_FILE_PATH, TOTAL_GPUS, PRIORITY_STRATEGY)

    # for strategy in ["critical_path", "max_concurrency", "exhaustive_combination"]:
    #     print("\n" + "="*80)
    #     print(f" E V A L U A T I N G   S T R A T E G Y :   {strategy.upper()} ")
    #     print("="*80 + "\n")
    #     evaluate_schedule_from_file(TASKS_FILE_PATH, TOTAL_GPUS, strategy)