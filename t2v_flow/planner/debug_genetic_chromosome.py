import json
import pandas as pd
from typing import List, Dict, Any, Tuple

# ==============================================================================
# ==============================================================================

def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    if k > len(gpu_timeline):
        return None, float('inf')
    sorted_gpus = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
    selected = sorted_gpus[:k]
    start = max(gpu_timeline[i] for i in selected)
    return selected, start

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
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

# ==============================================================================
# ==============================================================================

def generate_trace_from_chromosome_debug(tasks_input: List[Dict], chromosome: List[int], total_gpus: int) -> Tuple[List[Tuple], float]:
    print("--- 1. preprocessing ---")
    task_list_internal = []
    proc_times = {}
    sp_map = {}
    predecessors = {}

    chromosome_idx = 0
    for task_def in tasks_input:
        task_id = task_def['task_id']
        key_vae = (task_id, 'VAE')
        key_dit = (task_id, 'DIT')
        
        sp_vae_options = dict(task_def['vae'])
        sp_vae = chromosome[chromosome_idx]
        task_list_internal.append(key_vae)
        proc_times[key_vae] = sp_vae_options[sp_vae]
        sp_map[key_vae] = sp_vae
        chromosome_idx += 1
        
        sp_dit_options = dict(task_def['dit'])
        sp_dit = chromosome[chromosome_idx]
        task_list_internal.append(key_dit)
        proc_times[key_dit] = sp_dit_options[sp_dit]
        sp_map[key_dit] = sp_dit
        chromosome_idx += 1
        
        predecessors[key_dit] = [key_vae]
    
    print("--- 2. priority computation and ordering ---")
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)
    
    print("Task priority order (highest first):")
    for i, task in enumerate(task_list_internal):
        print(f"  {i+1}. {task}")
    
    ready_queue = list(task_list_internal)
    
    print("\n--- 3. scheduling loop ---")
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    loop_count = 0
    while len(completed_tasks) < len(proc_times):
        print(f"\n--- 🎬 While Loop Pass #{loop_count} ---")
        print(f"Current GPU timeline: {[f'{t:.2f}' for t in gpu_timeline]}")
        
        scheduled_this_loop = False
        
        for i, task_key in enumerate(list(ready_queue)):
            print(f"  -> checking task: {task_key} (position in the current queue: {i})")
            
            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                print(f"     [skip] dependencies not satisfied.")
                continue
            
            print(f"     dependencies satisfied.")
            k = sp_map[task_key]
            
            print(f"     calling find_earliest_gpus(timeline, k={k})...")
            gpus, start_time_gpu = find_earliest_gpus(gpu_timeline, k)
            print(f"     find_earliest_gpus returned: start_time_gpu = {start_time_gpu:.2f}, gpus = {gpus}")
            
            if gpus is None:
                print(f"     [skip] not enough GPUs.")
                continue

            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps], default=0)
            
            start = max(start_time_gpu, dep_finish_time)
            duration = proc_times[task_key]
            end = start + duration

            print(f"     computed schedule times:")
            print(f"        - GPU available at:    {start_time_gpu:.2f}")
            print(f"        - dependencies done at:{dep_finish_time:.2f}")
            print(f"        - final start time:    {start:.2f} (the later of the two)")
            
            print(f"     [scheduled] {task_key} from {start:.2f} to {end:.2f}")

            for g in gpus: gpu_timeline[g] = end
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            
            ready_queue.remove(task_key)
            
            scheduled_this_loop = True
            
            task_id, stage = task_key
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id, gpus))
            
            print("     --- break; starting the next while iteration ---")
            break

        loop_count += 1
        if not scheduled_this_loop and ready_queue:
            raise RuntimeError("Scheduling deadlock!")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


# ==============================================================================
# ==============================================================================
def debug_specific_chromosome(tasks_json_path: str, chromosome_to_debug: List[int], total_gpus: int):
    print(f"--- loading task file: {tasks_json_path} ---")
    try:
        with open(tasks_json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"[OK] loaded {len(tasks)} tasks")
    except Exception as e:
        print(f"[ERROR] failed to load the file: {e}")
        return

    print(f"--- debugging chromosome: {chromosome_to_debug} ---")
    print(f"--- total GPUs: {total_gpus} ---\n")

    final_trace, final_makespan = generate_trace_from_chromosome_debug(
        tasks_input=tasks,
        chromosome=chromosome_to_debug,
        total_gpus=total_gpus
    )
    
    print("\n\n" + "="*50)
    print("Scheduling complete")
    print("="*50)

    if not final_trace:
        print("[ERROR] the scheduler returned no valid trace.")
        return

    trace_df = pd.DataFrame(
        final_trace,
        columns=[
            "Task_ID", "Stage", "GPUs_Count", "Time",
            "Start", "End", "DatasetID", "GPU_List",
        ],
    ).sort_values(by=['Start', 'End']).reset_index(drop=True)

    print("\n--- final schedule trace (sorted by start time) ---")
    print(trace_df.to_string())
    
    print(f"\n\nFinal makespan = {final_makespan:.2f}s")


if __name__ == '__main__':
    
    TASKS_FILE_PATH = "generated_schedules/wan/schedule_4_tasks.json" 
    
    CHROMOSOME_TO_DEBUG = [4, 8, 10, 8, 4, 8, 4, 8]
    
    TOTAL_GPUS = 16

    debug_specific_chromosome(TASKS_FILE_PATH, CHROMOSOME_TO_DEBUG, TOTAL_GPUS)


