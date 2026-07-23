from collections import namedtuple

def flex_sp_schedule(tasks, total_gpus):
    
    task_cost_map = {}
    breakdown_map = {} 
    task_info_map = {}
    
    for task in tasks:
        t_id = task["task_id"]
        task_info_map[t_id] = task["dataset_id"]
        task_cost_map[t_id] = {}
        breakdown_map[t_id] = {}
        
        vae_times = {sp: t for sp, t in task["vae"]}
        
        for sp, dit_time in task["dit"]:
            vae_t = vae_times.get(sp, vae_times.get(1, 0)) 
            
            total_time = dit_time + vae_t
            
            task_cost_map[t_id][sp] = total_time
            breakdown_map[t_id][sp] = {'vae': vae_t, 'dit': dit_time}

    current_allocation = {} # task_id -> current_sp
    current_gpu_usage = 0
    
    for t_id, sp_options in task_cost_map.items():
        if not sp_options:
            print(f"❌ Task {t_id} has no valid SP options!")
            return [], 0
            
        min_sp = min(sp_options.keys())
        current_allocation[t_id] = min_sp
        current_gpu_usage += min_sp

    if current_gpu_usage > total_gpus:
        print(f"❌ OOM: Minimal config needs {current_gpu_usage} GPUs, but only {total_gpus} available.")
        return [], 0

    while current_gpu_usage < total_gpus:
        remaining_gpus = total_gpus - current_gpu_usage
        
        candidates = []
        for t_id, sp in current_allocation.items():
            time = task_cost_map[t_id][sp]
            candidates.append((time, t_id))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        improved = False
        
        for _, t_id in candidates:
            current_sp = current_allocation[t_id]
            available_sps = sorted(task_cost_map[t_id].keys())
            
            try:
                idx = available_sps.index(current_sp)
                if idx + 1 >= len(available_sps):
                    continue
                
                next_sp = available_sps[idx + 1]
                cost_diff = next_sp - current_sp
                
                if cost_diff <= remaining_gpus:
                    current_allocation[t_id] = next_sp
                    current_gpu_usage += cost_diff
                    improved = True
                    break
            except ValueError:
                continue
                
        if not improved:
            break

    trace = []
    gpu_cursor = 0
    max_makespan = 0
    
    base_start_time = 0.0
    
    for t_id in sorted(current_allocation.keys()):
        sp = current_allocation[t_id]
        
        times = breakdown_map[t_id][sp]
        vae_time = times['vae']
        dit_time = times['dit']
        total_time = vae_time + dit_time
        
        if total_time > max_makespan:
            max_makespan = total_time
            
        assigned_gpus = list(range(gpu_cursor, gpu_cursor + sp))
        gpu_cursor += sp
        
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