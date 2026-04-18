from collections import namedtuple

def flex_sp_schedule(tasks, total_gpus):
    """
    FlexSP 核心调度算法:
    1. 预处理: 将每个 Task 的 DiT 和 VAE 时间相加，整理出 {sp: total_time} 映射。
    2. 初始化: 给每个 Task 分配最小可用的 SP (Min SP)。
    3. 贪心优化: 循环找到当前最慢的任务，尝试给它升级 SP，直到 GPU 用完。
    4. 结果生成: 生成符合系统要求的 trace 列表 (VAE 和 DiT 分开显示)。
    """
    
    # --- 1. 数据预处理 & 成本映射 ---
    # task_cost_map[task_id][sp] = total_time (用于贪心算法比较)
    task_cost_map = {}
    # breakdown_map[task_id][sp] = {'vae': vae_time, 'dit': dit_time} (用于最后生成Trace)
    breakdown_map = {} 
    task_info_map = {} # 存储 dataset_id 等元数据
    
    for task in tasks:
        t_id = task["task_id"]
        task_info_map[t_id] = task["dataset_id"]
        task_cost_map[t_id] = {}
        breakdown_map[t_id] = {}
        
        # 将 VAE 时间转为字典方便查找
        vae_times = {sp: t for sp, t in task["vae"]}
        
        # 以 DiT 的 SP 列表为准 (因为 DiT 有内存限制)
        for sp, dit_time in task["dit"]:
            # 假设 VAE 和 DiT 使用相同的 SP
            # 如果 VAE 没有对应的 SP，则回退到 SP=1 (防止KeyError)
            vae_t = vae_times.get(sp, vae_times.get(1, 0)) 
            
            total_time = dit_time + vae_t
            
            # 存总时间用于优化
            task_cost_map[t_id][sp] = total_time
            # 存分项时间用于Trace
            breakdown_map[t_id][sp] = {'vae': vae_t, 'dit': dit_time}

    # --- 2. 初始化分配 (Min SP) ---
    current_allocation = {} # task_id -> current_sp
    current_gpu_usage = 0
    
    for t_id, sp_options in task_cost_map.items():
        if not sp_options:
            print(f"❌ Task {t_id} has no valid SP options!")
            return [], 0
            
        min_sp = min(sp_options.keys())
        current_allocation[t_id] = min_sp
        current_gpu_usage += min_sp

    # 检查资源是否足够启动
    if current_gpu_usage > total_gpus:
        print(f"❌ OOM: Minimal config needs {current_gpu_usage} GPUs, but only {total_gpus} available.")
        return [], 0

    # --- 3. 贪心优化 (Greedy Optimization) ---
    while current_gpu_usage < total_gpus:
        remaining_gpus = total_gpus - current_gpu_usage
        
        # a. 找出当前耗时最长的任务 (Critical Path)
        candidates = []
        for t_id, sp in current_allocation.items():
            time = task_cost_map[t_id][sp]
            candidates.append((time, t_id))
        
        # 按时间降序排列 (最慢的在前面)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        improved = False
        
        # b. 尝试升级最慢的任务 (如果不行就试次慢的 - Backfilling)
        for _, t_id in candidates:
            current_sp = current_allocation[t_id]
            available_sps = sorted(task_cost_map[t_id].keys())
            
            try:
                idx = available_sps.index(current_sp)
                if idx + 1 >= len(available_sps):
                    continue # 已经是最大 SP
                
                next_sp = available_sps[idx + 1]
                cost_diff = next_sp - current_sp
                
                # 检查是否有足够 GPU
                if cost_diff <= remaining_gpus:
                    current_allocation[t_id] = next_sp
                    current_gpu_usage += cost_diff
                    improved = True
                    break # 升级成功，重新计算 Critical Path
            except ValueError:
                continue
                
        if not improved:
            break # 无法再优化任何任务

    # --- 4. 生成 Trace (拆分 VAE 和 DiT) ---
    # Trace 格式参考: [Task_ID, Stage, GPUs_Count, Time, Start, End, DatasetID, GPU_List]
    trace = []
    gpu_cursor = 0 # 用于线性分配 GPU ID
    max_makespan = 0
    
    # 因为 FlexSP 是 Gang Scheduling (同步启动)，所有任务 Start 从 0 开始
    base_start_time = 0.0
    
    for t_id in sorted(current_allocation.keys()):
        sp = current_allocation[t_id]
        
        # 获取分项时间
        times = breakdown_map[t_id][sp]
        vae_time = times['vae']
        dit_time = times['dit']
        total_time = vae_time + dit_time
        
        # 记录最大 Makespan
        if total_time > max_makespan:
            max_makespan = total_time
            
        # 分配具体的 GPU ID (例如: [0,1,2,3])
        assigned_gpus = list(range(gpu_cursor, gpu_cursor + sp))
        gpu_cursor += sp
        
        # --- 生成 VAE 条目 ---
        # Start: 0
        # End: vae_time
        trace.append([
            t_id,                 # Task_ID
            "VAE",                # Stage
            sp,                   # GPUs_Count
            vae_time,             # Time (Duration)
            base_start_time,      # Start
            base_start_time + vae_time, # End
            task_info_map[t_id],  # DatasetID
            assigned_gpus         # GPU_List
        ])

        # --- 生成 DiT 条目 ---
        # Start: vae_time (紧接着 VAE 结束)
        # End: vae_time + dit_time
        trace.append([
            t_id,                 # Task_ID
            "DIT",                # Stage
            sp,                   # GPUs_Count
            dit_time,             # Time (Duration)
            base_start_time + vae_time, # Start
            base_start_time + total_time, # End
            task_info_map[t_id],  # DatasetID
            assigned_gpus         # GPU_List
        ])
        
    return trace, max_makespan