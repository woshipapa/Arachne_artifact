# def find_best_gpu_allocation_v2(
#     gpu_timeline: List[float], 
#     k: int, 
#     base_duration: float,
#     earliest_start_time: float,
#     lookahead_threshold: float = 20.0,
# ) -> Tuple[List[int], float, float, float]:
#     """


#     """
#     if k == 0:
#         return [], earliest_start_time, earliest_start_time, 1.0
    
#     potential_start_times = sorted(list(set(gpu_timeline)))
#     best_allocation = {
#         "gpus": None, "start_time": float('inf'),
#         "finish_time": float('inf'), "penalty": float('inf')
#     }

#     search_times = sorted(list(set([t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time])))

#     for t_start in search_times:
#         if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold:
#             break

#         available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]

#         if len(available_gpus) < k:
#             continue

#         try:
#             import math
#             num_combinations = math.comb(len(available_gpus), k)
#         except ValueError:
#             num_combinations = 0

#         if num_combinations > max_combinations_to_check:
#             all_combinations_iterator = itertools.combinations(available_gpus, k)
#             combinations_to_check = random.sample(list(all_combinations_iterator), max_combinations_to_check)
#             total_to_check = max_combinations_to_check
#         else:
#             combinations_to_check = itertools.combinations(available_gpus, k)
#             total_to_check = num_combinations
        
#         if total_to_check > 50: 
#             pbar = tqdm(combinations_to_check, total=total_to_check,
#                         desc=f"      Search @ t={t_start:<5.1f}s ({len(available_gpus)}C{k})",
#                         ncols=120,
#                         unit=" combo",
#                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
#         else:

#         for gpu_combination in pbar:
#             gpu_list = list(gpu_combination)
#             penalty = topology_model.predict_penalty(gpu_list)
#             actual_duration = base_duration * penalty
#             finish_time = t_start + actual_duration
            
#             if finish_time < best_allocation["finish_time"]:
#                 best_allocation = {
#                     "gpus": gpu_list, "start_time": t_start,
#                     "finish_time": finish_time, "penalty": penalty
#                 }

#     if best_allocation["gpus"] is None:
#         sorted_gpus_by_time = sorted(enumerate(gpu_timeline), key=lambda x: x[1])
#         gpus_to_wait_for = [g[0] for g in sorted_gpus_by_time[:k]]
#         forced_start_time = max(gpu_timeline[g] for g in gpus_to_wait_for)
        
#         penalty = topology_model.predict_penalty(gpus_to_wait_for)
#         actual_duration = base_duration * penalty
#         finish_time = forced_start_time + actual_duration
        
#         print(f"      Fallback: No immediate solution found. Waiting for {k} earliest GPUs.")
#         return gpus_to_wait_for, forced_start_time, finish_time, penalty

#     return (
#         best_allocation["gpus"], 
#         best_allocation["start_time"], 
#         best_allocation["finish_time"], 
#         best_allocation["penalty"]
#     )


# def find_best_gpu_allocation(
#     gpu_timeline: List[float], 
#     k: int, 
#     base_duration: float,
#     earliest_start_time: float,
#     topology_model: TopologyModel,
# ) -> Tuple[List[int], float, float, float]:
#     """

#     """
#     if k == 0:
#         return [], earliest_start_time, earliest_start_time, 1.0
    
#     potential_start_times = sorted(list(set(gpu_timeline)))
#     search_start_time = earliest_start_time
    
#     best_allocation = {
#         "gpus": None,
#         "start_time": float('inf'),
#         "finish_time": float('inf'),
#         "penalty": float('inf')
#     }

#     for t_start in [t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time]:
        
#         if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold:
#             break

#         available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]

#         if len(available_gpus) < k:

#         for gpu_combination in itertools.combinations(available_gpus, k):
#             gpu_list = list(gpu_combination)
            
#             penalty = topology_model.predict_penalty(gpu_list)
#             actual_duration = base_duration * penalty
            
#             finish_time = t_start + actual_duration
            
#             if finish_time < best_allocation["finish_time"]:
#                 best_allocation = {
#                     "gpus": gpu_list,
#                     "start_time": t_start,
#                     "finish_time": finish_time,
#                     "penalty": penalty
#                 }

#     if best_allocation["gpus"] is None:
#         sorted_gpus_by_time = sorted(enumerate(gpu_timeline), key=lambda x: x[1])
#         gpus_to_wait_for = [g[0] for g in sorted_gpus_by_time[:k]]
#         forced_start_time = max(gpu_timeline[g] for g in gpus_to_wait_for)
        
#         penalty = topology_model.predict_penalty(gpus_to_wait_for)
#         actual_duration = base_duration * penalty
#         finish_time = forced_start_time + actual_duration

#         return gpus_to_wait_for, forced_start_time, finish_time, penalty


#     return (
#         best_allocation["gpus"], 
#         best_allocation["start_time"], 
#         best_allocation["finish_time"], 
#         best_allocation["penalty"]
#     )


# def generate_trace_from_chromosome(tasks_input: List[Dict], chromosome: List[int], total_gpus: int) -> Tuple[List[Tuple], float]:
#     """
#     """
#     task_list_internal = []
#     proc_times = {}
#     sp_map = {}
#     predecessors = {}

#     chromosome_idx = 0
#     for task_def in tasks_input:
#         task_id = task_def['task_id']
#         key_vae = (task_id, 'VAE')
#         key_dit = (task_id, 'DIT')
        
#         sp_vae = chromosome[chromosome_idx]
#         task_list_internal.append(key_vae)
#         proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
#         sp_map[key_vae] = sp_vae
#         chromosome_idx += 1
        
#         sp_dit = chromosome[chromosome_idx]
#         task_list_internal.append(key_dit)
#         proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
#         sp_map[key_dit] = sp_dit
#         chromosome_idx += 1
        
#         predecessors[key_dit] = [key_vae]

#     gpu_timeline = [0.0] * total_gpus
#     trace = []
#     stage_end_time = {}
#     completed_tasks = set()
    
#     get_critical_path_priority.cache = {}
#     task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

#     while len(completed_tasks) < len(proc_times):
#         scheduled_this_loop = False
#             if task_key in completed_tasks: continue

#             deps = predecessors.get(task_key, [])
#             if not all(dep in completed_tasks for dep in deps):
#                 continue
            
#             k = sp_map[task_key]
#             gpus, start_time_gpu = find_earliest_gpus(gpu_timeline, k)
            
#             if gpus is None: continue

#             dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps]) if deps else 0
            
#             start = max(start_time_gpu, dep_finish_time)
#             duration = proc_times[task_key]
#             end = start + duration

#             for g in gpus: gpu_timeline[g] = end
            
#             task_id, stage = task_key
#             dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
#             trace.append((task_id, stage, k, duration, start, end, dataset_id, gpus))
            
#             stage_end_time[task_key] = end
#             completed_tasks.add(task_key)
#             task_list_internal.remove(task_key)
#             scheduled_this_loop = True
        
#         if not scheduled_this_loop and task_list_internal:
            
#     makespan = max(gpu_timeline) if gpu_timeline else 0
#     return trace, makespan
