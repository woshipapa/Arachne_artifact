# # 随机采样k个GPU的组合
# def find_best_gpu_allocation_v2(
#     gpu_timeline: List[float], 
#     k: int, 
#     base_duration: float,
#     earliest_start_time: float,
#     topology_model: 'TopologyModel', # 假设 TopologyModel 已定义
#     lookahead_threshold: float = 20.0,
#     max_combinations_to_check: int = 1000 # 新增：剪枝阈值
# ) -> Tuple[List[int], float, float, float]:
#     """
#     【V2 - 带剪枝和进度条】拓扑感知的GPU分配函数。
#     寻找能让任务“最早结束”的GPU分配方案。

#     新增功能:
#     - 组合剪枝: 当可能的GPU组合数超过阈值时，进行随机采样，避免计算过久。
#     - TQDM进度条: 在搜索大量组合时，显示实时进度。

#     返回: (最佳GPU列表, 最佳开始时间, 最佳结束时间, 应用的惩罚系数)
#     """
#     if k == 0:
#         return [], earliest_start_time, earliest_start_time, 1.0
    
#     potential_start_times = sorted(list(set(gpu_timeline)))
#     best_allocation = {
#         "gpus": None, "start_time": float('inf'),
#         "finish_time": float('inf'), "penalty": float('inf')
#     }

#     # 遍历所有未来GPU释放的“事件点”作为潜在的开始时间
#     # 将 earliest_start_time 也加入搜索列表，以应对依赖任务刚结束但所有GPU都在忙的情况
#     search_times = sorted(list(set([t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time])))

#     for t_start in search_times:
#         if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold:
#             break

#         available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]

#         if len(available_gpus) < k:
#             continue

#         # --- 剪枝策略 ---
#         try:
#             import math
#             num_combinations = math.comb(len(available_gpus), k)
#         except ValueError:
#             num_combinations = 0

#         if num_combinations > max_combinations_to_check:
#             # 组合数过多，进行随机采样
#             all_combinations_iterator = itertools.combinations(available_gpus, k)
#             # 为了高效采样，需要先将迭代器物化为列表（可能会消耗一些内存）
#             combinations_to_check = random.sample(list(all_combinations_iterator), max_combinations_to_check)
#             total_to_check = max_combinations_to_check
#         else:
#             combinations_to_check = itertools.combinations(available_gpus, k)
#             total_to_check = num_combinations
        
#         # --- TQDM 进度条 ---
#         # 只有在组合数较多时才显示进度条，避免频繁刷新造成干扰
#         if total_to_check > 50: 
#             pbar = tqdm(combinations_to_check, total=total_to_check,
#                         desc=f"      Search @ t={t_start:<5.1f}s ({len(available_gpus)}C{k})",
#                         leave=False, # 结束后自动清除进度条
#                         ncols=120,
#                         unit=" combo",
#                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
#         else:
#             pbar = combinations_to_check # 组合数少，直接迭代

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

#     # --- 回退逻辑 (Fallback) ---
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




# # 按照时间完成最早的k个，更简略
# def find_best_gpu_allocation(
#     gpu_timeline: List[float], 
#     k: int, 
#     base_duration: float,
#     earliest_start_time: float,
#     topology_model: TopologyModel,
#     lookahead_threshold: float = 5  # 定义“耐心”阈值，单位：秒
# ) -> Tuple[List[int], float, float, float]:
#     """
#     【全新】拓扑感知的GPU分配函数。
#     寻找能让任务“最早结束”的GPU分配方案。

#     返回: (最佳GPU列表, 最佳开始时间, 最佳结束时间, 应用的惩罚系数)
#     """
#     if k == 0:
#         return [], earliest_start_time, earliest_start_time, 1.0
    
#     # 1. 确定搜索的时间窗口
#     # 找到当前可用的GPU的最早释放时间作为搜索起点
#     potential_start_times = sorted(list(set(gpu_timeline)))
#     search_start_time = earliest_start_time
    
#     # 2. 初始化最佳方案记录
#     best_allocation = {
#         "gpus": None,
#         "start_time": float('inf'),
#         "finish_time": float('inf'),
#         "penalty": float('inf')
#     }

#     # 3. 在时间窗口内搜索
#     # 遍历所有未来GPU释放的“事件点”作为潜在的开始时间
#     for t_start in [t for t in potential_start_times if t >= earliest_start_time] + [earliest_start_time]:
        
#         # 超过“耐心”阈值后，如果已经有方案，就不再搜索，避免无尽等待
#         if best_allocation["gpus"] and t_start > best_allocation["start_time"] + lookahead_threshold:
#             break

#         # 获取在 t_start 这个时间点可用的所有GPU
#         available_gpus = [i for i, free_time in enumerate(gpu_timeline) if free_time <= t_start]

#         if len(available_gpus) < k:
#             continue # 当前时间点可用GPU不足，跳到下一个事件点

#         # 4. 评估所有可能的GPU组合
#         # 从可用的GPU中，找出所有大小为k的组合
#         for gpu_combination in itertools.combinations(available_gpus, k):
#             gpu_list = list(gpu_combination)
            
#             # 计算通信惩罚和实际执行时间
#             penalty = topology_model.predict_penalty(gpu_list)
#             actual_duration = base_duration * penalty
            
#             # 计算任务的完成时间
#             finish_time = t_start + actual_duration
            
#             # 5. 更新最优解
#             # 如果当前组合的完成时间更早，就更新为最佳方案
#             if finish_time < best_allocation["finish_time"]:
#                 best_allocation = {
#                     "gpus": gpu_list,
#                     "start_time": t_start,
#                     "finish_time": finish_time,
#                     "penalty": penalty
#                 }

#     if best_allocation["gpus"] is None:
#         # 如果遍历完所有当前和未来的“事件点”都找不到可用资源，说明可能需要等待某个正在运行的任务结束
#         # 此时，我们退化为找到k个最快完成的任务，并在它们结束后立刻开始
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
#     【桥梁函数】
#     接收GA找到的最优SP组合(chromosome)，并使用高效的列表调度算法生成最终的trace。
#     这是连接GA和你的YAML生成器的关键。
#     """
#     # a. 数据预处理，以适配列表调度器
#     task_list_internal = []
#     proc_times = {}
#     sp_map = {}
#     predecessors = {}

#     chromosome_idx = 0
#     for task_def in tasks_input:
#         task_id = task_def['task_id']
#         key_vae = (task_id, 'VAE')
#         key_dit = (task_id, 'DIT')
        
#         # VAE 任务信息
#         sp_vae = chromosome[chromosome_idx]
#         task_list_internal.append(key_vae)
#         proc_times[key_vae] = dict(task_def['vae'])[sp_vae]
#         sp_map[key_vae] = sp_vae
#         chromosome_idx += 1
        
#         # DiT 任务信息
#         sp_dit = chromosome[chromosome_idx]
#         task_list_internal.append(key_dit)
#         proc_times[key_dit] = dict(task_def['dit'])[sp_dit]
#         sp_map[key_dit] = sp_dit
#         chromosome_idx += 1
        
#         # 依赖关系
#         predecessors[key_dit] = [key_vae]

#     # b. 列表调度器核心逻辑
#     gpu_timeline = [0.0] * total_gpus
#     trace = []
#     stage_end_time = {}
#     completed_tasks = set()
    
#     # 使用关键路径作为任务优先级
#     get_critical_path_priority.cache = {}
#     task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)

#     while len(completed_tasks) < len(proc_times):
#         scheduled_this_loop = False
#         for task_key in list(task_list_internal): # 遍历副本
#             if task_key in completed_tasks: continue

#             # 检查所有前置依赖是否已完成
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
#             raise RuntimeError("调度死锁！请检查资源和依赖。")
            
#     makespan = max(gpu_timeline) if gpu_timeline else 0
#     return trace, makespan
