import heapq
import random
import time
import json
import os
from typing import List, Dict, Tuple, Set, Optional
from itertools import combinations


class ClusterTopology:
    def __init__(self, total_gpus: int, nodes: int, gpus_per_node: int, nvlink_groups: List[Set[int]]):
        self.total_gpus = total_gpus
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.nvlink_groups = nvlink_groups
        
        self.gpu_to_node = {i: i // gpus_per_node for i in range(total_gpus)}
        self.gpu_to_nvlink_group = {}
        for i, group in enumerate(nvlink_groups):
            for gpu in group:
                self.gpu_to_nvlink_group[gpu] = i

    def get_gpu_group_compactness_score(self, gpus: List[int]) -> int:
        if not gpus: return 2
        nodes = {self.gpu_to_node[g] for g in gpus}
        if len(nodes) > 1: return 2
        
        nvlink_groups = {self.gpu_to_nvlink_group.get(g) for g in gpus}
        if len(nvlink_groups) == 1 and None not in nvlink_groups: return 0
        
        return 1

    def get_inter_group_distance_score(self, gpus1: List[int], gpus2: List[int]) -> int:
        if not gpus1 or not gpus2: return 1
        nodes1 = {self.gpu_to_node[g] for g in gpus1}
        nodes2 = {self.gpu_to_node[g] for g in gpus2}
        if not nodes1.isdisjoint(nodes2): return 0
        return 1

COMMUNICATION_COSTS = {
    "BROADCAST": {0: 0.5, 1: 1.5, 2: 4.0},
    "TRANSFER": {0: 2.0, 1: 8.0}
}

def load_tasks_from_json(filepath: str) -> List[Dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
        print(f"Loaded {len(tasks_data)} tasks from {filepath}.")
        return tasks_data
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}.")
        return []
    except json.JSONDecodeError:
        print(f"Error: failed to parse the JSON file {filepath}.")
        return []


def generate_trace_agnostic(chromosome: List[int], tasks: List[Dict], total_gpus: int) -> float:
    num_tasks = len(tasks)
    sp_config = {f"{i}_VAE": tasks[i]["vae"][chromosome[i]] for i in range(num_tasks)}
    sp_config.update({f"{i}_DIT": tasks[i]["dit"][chromosome[num_tasks + i]] for i in range(num_tasks)})

    gpu_timeline = [0.0] * total_gpus
    stage_end_times = {}
    ready_queue = [f"{i}_VAE" for i in range(num_tasks)]
    processed_tasks = set()
    
    while ready_queue:
        task_key = ready_queue.pop(0)
        task_id_str, stage = task_key.split('_')
        task_id = int(task_id_str)
        k, t = sp_config[task_key]

        available_times = heapq.nsmallest(k, gpu_timeline)
        gpu_ready_time = max(available_times) if available_times else 0.0
        
        start_time = gpu_ready_time
        if stage == "DIT":
            vae_key = f"{task_id}_VAE"
            start_time = max(start_time, stage_end_times[vae_key])

        end_time = start_time + t
        
        indices_to_update = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])[:k]
        for idx in indices_to_update: gpu_timeline[idx] = end_time
        stage_end_times[task_key] = end_time
        processed_tasks.add(task_key)

        if stage == "VAE": ready_queue.append(f"{task_id}_DIT")
    return max(gpu_timeline) if gpu_timeline else 0.0

def generate_trace_aware_penalty(chromosome: List[int], tasks: List[Dict], total_gpus: int) -> float:
    num_tasks = len(tasks)
    sp_config = {f"{i}_VAE": tasks[i]["vae"][chromosome[i]] for i in range(num_tasks)}
    sp_config.update({f"{i}_DIT": tasks[i]["dit"][chromosome[num_tasks + i]] for i in range(num_tasks)})

    gpu_timeline = [0.0] * total_gpus
    stage_end_times = {}
    ready_queue = [f"{i}_VAE" for i in range(num_tasks)]
    processed_tasks = set()
    
    PARTIAL_OVERLAP_PENALTY_FACTOR, NO_OVERLAP_PENALTY_FACTOR = 0.1, 0.25

    while ready_queue:
        task_key = ready_queue.pop(0)
        task_id_str, stage = task_key.split('_')
        task_id = int(task_id_str)
        k, t = sp_config[task_key]

        available_times = heapq.nsmallest(k, gpu_timeline)
        gpu_ready_time = max(available_times) if available_times else 0.0
        
        start_time = gpu_ready_time
        if stage == "DIT":
            vae_key = f"{task_id}_VAE"
            vae_sp, _ = sp_config[vae_key]
            dit_sp = k
            comm_penalty = 0
            if vae_sp != dit_sp:
                sp_diff = abs(vae_sp - dit_sp)
                comm_penalty = t * sp_diff * (PARTIAL_OVERLAP_PENALTY_FACTOR if dit_sp < total_gpus / 2 else NO_OVERLAP_PENALTY_FACTOR)
            start_time = max(start_time, stage_end_times[vae_key]) + comm_penalty

        end_time = start_time + t
        
        indices_to_update = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])[:k]
        for idx in indices_to_update: gpu_timeline[idx] = end_time
        stage_end_times[task_key] = end_time
        processed_tasks.add(task_key)

        if stage == "VAE": ready_queue.append(f"{task_id}_DIT")
    return max(gpu_timeline) if gpu_timeline else 0.0


def create_greedy_chromosome(tasks: List[Dict]) -> List[int]:
    chromosome = []
    num_tasks = len(tasks)
    for i in range(num_tasks):
        fastest_vae_idx = min(range(len(tasks[i]["vae"])), key=lambda j: tasks[i]["vae"][j][1])
        chromosome.append(fastest_vae_idx)
    for i in range(num_tasks):
        fastest_dit_idx = min(range(len(tasks[i]["dit"])), key=lambda j: tasks[i]["dit"][j][1])
        chromosome.append(fastest_dit_idx)
    return chromosome

def run_genetic_algorithm(tasks: List[Dict], total_gpus: int, fitness_func, generations=50, pop_size=30, mutation_rate=0.1, crossover_rate=0.8):
    num_tasks = len(tasks)
    
    population = []
    greedy_seed = create_greedy_chromosome(tasks)
    population.append(greedy_seed)
    
    while len(population) < pop_size:
        chromosome = [random.randint(0, len(tasks[i]["vae"]) - 1) for i in range(num_tasks)]
        chromosome.extend([random.randint(0, len(tasks[i]["dit"]) - 1) for i in range(num_tasks)])
        population.append(chromosome)

    best_overall_chromosome, best_overall_fitness = None, float('inf')

    for gen in range(generations):
        fitness_scores = [fitness_func(c, tasks, total_gpus) for c in population]
        
        min_fitness_in_gen = min(fitness_scores)
        if min_fitness_in_gen < best_overall_fitness:
            best_overall_fitness = min_fitness_in_gen
            best_overall_chromosome = population[fitness_scores.index(min_fitness_in_gen)]
        
        if (gen + 1) % 20 == 0:
            print(f"  gen {gen+1}/{generations}, best makespan this generation (est.): {min_fitness_in_gen:.2f}, global best (est.): {best_overall_fitness:.2f}")

        new_population = []
        elite_index = fitness_scores.index(min(fitness_scores))
        new_population.append(population[elite_index])

        while len(new_population) < pop_size:
            if all(f == 0 for f in fitness_scores): parents = [random.choice(population), random.choice(population)]
            else:
                total_fitness = sum(1 / f for f in fitness_scores if f > 0)
                parents = []
                for _ in range(2):
                    pick = random.uniform(0, total_fitness)
                    current = 0
                    for i, score in enumerate(fitness_scores):
                        if score > 0: current += 1 / score
                        if current > pick:
                            parents.append(population[i]); break
                if len(parents) < 2: parents.append(random.choice(population))
            
            parent1, parent2 = parents[0], parents[1]
            if random.random() < crossover_rate:
                point = random.randint(1, len(parent1) - 1)
                child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
            else:
                child1, child2 = parent1[:], parent2[:]

            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    gene_to_mutate = random.randint(0, len(child) - 1)
                    task_idx = gene_to_mutate % num_tasks
                    stage_options = "vae" if gene_to_mutate < num_tasks else "dit"
                    child[gene_to_mutate] = random.randint(0, len(tasks[task_idx][stage_options]) - 1)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]

    return best_overall_chromosome, best_overall_fitness


def get_critical_path_priority(task_key: str, predecessors: Dict, proc_times: Dict, cache: Dict) -> float:
    if task_key in cache: return cache[task_key]
    successors = [t for t, preds in predecessors.items() if task_key in preds]
    if not successors: return proc_times[task_key]
    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times, cache) for succ in successors)
    result = proc_times[task_key] + max_succ_path
    cache[task_key] = result
    return result

def find_best_placement_for_dit(gpu_timeline: List[float], k: int, t: float, topology: ClusterTopology, vae_end_time: float, prev_gpus: List[int]) -> Tuple[Optional[List[int]], float, float, str]:
    best_gpus, best_end_time, best_start_time, best_strategy = None, float('inf'), float('inf'), "N/A"

    if k == len(prev_gpus):
        ready_time = max(gpu_timeline[g] for g in prev_gpus)
        start = max(ready_time, vae_end_time)
        end = start + t
        best_end_time, best_start_time, best_gpus, best_strategy = end, start, prev_gpus, "full overlap (zero cost)"

    search_space = list(combinations(range(topology.total_gpus), k))
    random.shuffle(search_space)
    
    for gpus_tuple in search_space[:min(len(search_space), 2000)]:
        gpus = list(gpus_tuple)
        if set(gpus) == set(prev_gpus): continue
        ready_time = max(gpu_timeline[g] for g in gpus)
        comm_cost, strategy = 0, ""
        overlap = set(gpus) & set(prev_gpus)
        if not overlap:
            dist = topology.get_inter_group_distance_score(prev_gpus, gpus)
            comm_cost = COMMUNICATION_COSTS["TRANSFER"][dist]
            strategy = f"no overlap (transfer cost: {comm_cost})"
        else:
            comp = topology.get_gpu_group_compactness_score(gpus)
            comm_cost = COMMUNICATION_COSTS["BROADCAST"][comp]
            strategy = f"partial overlap (broadcast cost: {comm_cost})"
        start = max(ready_time, vae_end_time) + comm_cost
        end = start + t
        if end < best_end_time:
            best_end_time, best_start_time, best_gpus, best_strategy = end, start, gpus, strategy
    return best_gpus, best_start_time, best_end_time, best_strategy

def find_best_placement_for_vae(gpu_timeline: List[float], k: int, t: float, topology: ClusterTopology):
    best_gpus, best_end_time, best_compactness = None, float('inf'), float('inf')
    search_space = list(combinations(range(topology.total_gpus), k))
    random.shuffle(search_space)
    for gpus_tuple in search_space[:min(len(search_space), 2000)]:
        gpus = list(gpus_tuple)
        start = max(gpu_timeline[g] for g in gpus)
        end = start + t
        compactness = topology.get_gpu_group_compactness_score(gpus)
        if compactness < best_compactness or (compactness == best_compactness and end < best_end_time):
            best_compactness, best_end_time, best_gpus = compactness, end, gpus
    start = best_end_time - t
    return best_gpus, start, best_end_time, f"compactness first (compactness: {best_compactness})"

def topology_aware_physical_allocator_stage(logical_plan: List[int], tasks: List[Dict], topology: ClusterTopology):
    print("\n--- stage 3: physical assignment with the topology-aware allocator ---")
    num_tasks = len(tasks)
    
    sp_config = {f"{i}_VAE": tasks[i]["vae"][logical_plan[i]] for i in range(num_tasks)}
    sp_config.update({f"{i}_DIT": tasks[i]["dit"][logical_plan[num_tasks + i]] for i in range(num_tasks)})
    
    proc_times = {k: v[1] for k, v in sp_config.items()}
    predecessors = {f"{i}_DIT": {f"{i}_VAE"} for i in range(num_tasks)}

    gpu_timeline = [0.0] * topology.total_gpus
    stage_end_times, task_to_gpus = {}, {}
    ready_queue = {f"{i}_VAE" for i in range(num_tasks)}
    final_schedule, processed_tasks = [], set()
    
    path_cache = {}
    
    while len(processed_tasks) < 2 * num_tasks:
        if not ready_queue: break
        
        sorted_ready = sorted(list(ready_queue), key=lambda t: get_critical_path_priority(t, predecessors, proc_times, path_cache), reverse=True)
        
        task_key = sorted_ready[0]
        
        task_id_str, stage = task_key.split('_')
        k, t = sp_config[task_key]
        
        if stage == "VAE":
            gpus, start, end, strategy = find_best_placement_for_vae(gpu_timeline, k, t, topology)
        else:
            vae_key = f"{task_id_str}_VAE"
            gpus, start, end, strategy = find_best_placement_for_dit(gpu_timeline, k, t, topology, stage_end_times[vae_key], task_to_gpus[vae_key])
        
        if gpus is None:
            print(f"Warning: task {task_key} cannot be placed yet; retrying next round.")
            continue

        ready_queue.remove(task_key)
        print(f"  [decision] task: {task_key:<8} | strategy: {strategy}")

        for g in gpus: gpu_timeline[g] = end
        stage_end_times[task_key] = end
        task_to_gpus[task_key] = gpus
        processed_tasks.add(task_key)
        
        final_schedule.append({"task": task_key, "sp": k, "time": t, "start": start, "end": end, "gpus": gpus, "strategy": strategy})
        
        for i in range(num_tasks):
            dit_key = f"{i}_DIT"
            if dit_key not in processed_tasks and dit_key not in ready_queue:
                vae_key = f"{i}_VAE"
                if vae_key in processed_tasks:
                    ready_queue.add(dit_key)

    final_makespan = max(gpu_timeline) if gpu_timeline else 0
    print(f"--- stage 3: physical assignment complete ---")
    
    print("\nDetailed schedule trace:")
    for item in sorted(final_schedule, key=lambda x: x['start']):
        print(f"  - task: {item['task']:<8} | SP: {item['sp']} | start: {item['start']:.2f} | end: {item['end']:.2f} | GPUs: {sorted(item['gpus'])}")
              
    return final_makespan, final_schedule

if __name__ == "__main__":
    TOTAL_GPUS = 16
    NODES = 2
    GPUS_PER_NODE = 8
    NVLINK_GROUPS = [set(range(i, i + GPUS_PER_NODE)) for i in range(0, TOTAL_GPUS, GPUS_PER_NODE)]
    
    topology = ClusterTopology(TOTAL_GPUS, NODES, GPUS_PER_NODE, NVLINK_GROUPS)
    
    iteration_number = 4 
    tasks_filepath = f"generated_schedules/wan/schedule_{iteration_number}_tasks.json"
    
    if not os.path.exists(tasks_filepath):
        print(f"Warning: task file {tasks_filepath} not found. Creating a mock file for the demo.")
        os.makedirs(os.path.dirname(tasks_filepath), exist_ok=True)
        mock_tasks_data = [
            {"task_id": 0, "dataset_id": "1_101_1280_720_rank0", "dit": [[4, 30.15], [5, 22.67], [8, 13.97], [10, 12.60]], "vae": [[1, 7.38], [2, 3.87], [4, 2.27], [5, 1.95], [8, 1.48], [10, 1.17]]},
            {"task_id": 1, "dataset_id": "2_45_1280_720_rank2", "dit": [[4, 16.38], [5, 11.74], [8, 7.45], [10, 7.22]], "vae": [[1, 5.70], [2, 2.96], [4, 1.77], [5, 1.50], [8, 1.13], [10, 0.88]]},
            {"task_id": 2, "dataset_id": "1_113_1280_720_rank4", "dit": [[4, 35.69], [5, 28.46], [8, 17.26], [10, 14.96]], "vae": [[1, 8.29], [2, 4.28], [4, 2.53], [5, 2.19], [8, 1.67], [10, 1.32]]},
            {"task_id": 3, "dataset_id": "1_113_1280_720_rank6", "dit": [[4, 35.69], [5, 28.46], [8, 17.26], [10, 14.96]], "vae": [[1, 8.29], [2, 4.28], [4, 2.53], [5, 2.19], [8, 1.67], [10, 1.32]]}
        ]
        with open(tasks_filepath, 'w', encoding='utf-8') as f: json.dump(mock_tasks_data, f, indent=4)

    tasks = load_tasks_from_json(tasks_filepath)
    if not tasks: exit("Task loading failed; exiting.")

    GA_PARAMS = {"generations": 200, "pop_size": 200, "mutation_rate": 0.1, "crossover_rate": 0.8}

    print("\n--- stage 1: topology-agnostic GA ---")
    _, agnostic_makespan = run_genetic_algorithm(tasks, TOTAL_GPUS, generate_trace_agnostic, **GA_PARAMS)
    
    print("\n--- stage 2: topology-aware GA (with informed initialisation) ---")
    aware_plan, aware_estimated_makespan = run_genetic_algorithm(tasks, TOTAL_GPUS, generate_trace_aware_penalty, **GA_PARAMS)
    
    if not aware_plan: exit("The topology-aware GA found no valid schedule; exiting.")
    
    print("\nBest logical plan found by the topology-aware GA (SP configuration):")
    for i in range(len(tasks)):
        vae_sp_idx, dit_sp_idx = aware_plan[i], aware_plan[len(tasks) + i]
        print(f"  task {i}: VAE SP={tasks[i]['vae'][vae_sp_idx][0]}, DIT SP={tasks[i]['dit'][dit_sp_idx][0]}")

    final_makespan, _ = topology_aware_physical_allocator_stage(aware_plan, tasks, topology)

    print("\n" + "="*50)
    print("[Final comparison]")
    print(f"  1. theoretical best (compute only, no communication cost): {agnostic_makespan:.2f}s")
    print(f"  2. GA estimate (with the communication penalty):           {aware_estimated_makespan:.2f}s")
    print(f"  3. final physical assignment (exact communication cost):   {final_makespan:.2f}s")
    print("="*50)
