import json
import pandas as pd
from typing import List, Dict, Any, Tuple

# ==============================================================================
# 核心函数 (来自您的代码)
# ==============================================================================

def find_earliest_gpus(gpu_timeline: List[float], k: int) -> Tuple[List[int], float]:
    """您的 find_earliest_gpus 函数"""
    if k > len(gpu_timeline):
        return None, float('inf')
    sorted_gpus = sorted(range(len(gpu_timeline)), key=lambda i: gpu_timeline[i])
    selected = sorted_gpus[:k]
    start = max(gpu_timeline[i] for i in selected)
    return selected, start

def get_critical_path_priority(task_key: Tuple[int, str], predecessors: Dict, proc_times: Dict) -> float:
    """您的 get_critical_path_priority 函数"""
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
# 带有调试信息的核心调度函数
# ==============================================================================

def generate_trace_from_chromosome_debug(tasks_input: List[Dict], chromosome: List[int], total_gpus: int) -> Tuple[List[Tuple], float]:
    """
    这是一个带有详细打印语句的、用于调试的调度函数版本。
    """
    print("--- ⚙️ 1. 数据预处理开始 ---")
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
    
    print("--- 📊 2. 优先级计算与排序 ---")
    get_critical_path_priority.cache = {}
    task_list_internal.sort(key=lambda t: get_critical_path_priority(t, predecessors, proc_times), reverse=True)
    
    # 打印出完整的优先级列表
    print("任务优先级排序 (从高到低):")
    for i, task in enumerate(task_list_internal):
        print(f"  {i+1}. {task}")
    
    ready_queue = list(task_list_internal)
    
    print("\n--- 🚀 3. 调度循环开始 ---")
    gpu_timeline = [0.0] * total_gpus
    trace = []
    stage_end_time = {}
    completed_tasks = set()
    
    loop_count = 0
    while len(completed_tasks) < len(proc_times):
        print(f"\n--- 🎬 While Loop Pass #{loop_count} ---")
        print(f"当前GPU时间线: {[f'{t:.2f}' for t in gpu_timeline]}")
        
        scheduled_this_loop = False
        
        # 这里用 `for i, task_key in enumerate(list(ready_queue))` 来安全遍历
        for i, task_key in enumerate(list(ready_queue)):
            print(f"  -> 检查任务: {task_key} (在当前队列中的位置: {i})")
            
            deps = predecessors.get(task_key, [])
            if not all(dep in completed_tasks for dep in deps):
                print(f"     ❌ 依赖未满足. 跳过.")
                continue
            
            print(f"     ✅ 依赖已满足.")
            k = sp_map[task_key]
            
            # 调试 find_earliest_gpus 的关键步骤
            print(f"     📞 调用 find_earliest_gpus(timeline, k={k})...")
            gpus, start_time_gpu = find_earliest_gpus(gpu_timeline, k)
            print(f"     📞 find_earliest_gpus 返回: start_time_gpu = {start_time_gpu:.2f}, gpus = {gpus}")
            
            if gpus is None:
                print(f"     ❌ 没有足够的GPU. 跳过.")
                continue

            dep_finish_time = max([stage_end_time.get(dep, 0) for dep in deps], default=0)
            
            start = max(start_time_gpu, dep_finish_time)
            duration = proc_times[task_key]
            end = start + duration

            print(f"     ✨ 计算出的调度时间:")
            print(f"        - GPU可用开始时间: {start_time_gpu:.2f}")
            print(f"        - 依赖完成时间:    {dep_finish_time:.2f}")
            print(f"        - 最终开始时间:    {start:.2f} (取二者最大值)")
            
            print(f"     🚀 [成功] 调度 {task_key} 从 {start:.2f} 到 {end:.2f}")

            # 更新状态
            for g in gpus: gpu_timeline[g] = end
            stage_end_time[task_key] = end
            completed_tasks.add(task_key)
            
            # 从原始队列中按值移除
            ready_queue.remove(task_key)
            
            scheduled_this_loop = True
            
            task_id, stage = task_key
            dataset_id = next(t['dataset_id'] for t in tasks_input if t['task_id'] == task_id)
            trace.append((task_id, stage.upper(), k, duration, start, end, dataset_id, gpus))
            
            print("     --- 循环中断 (break), 准备开始下一轮 While 循环 ---")
            break

        loop_count += 1
        if not scheduled_this_loop and ready_queue:
            raise RuntimeError("调度死锁！")
            
    makespan = max(gpu_timeline) if gpu_timeline else 0
    return trace, makespan


# ==============================================================================
# 主调试流程
# ==============================================================================
def debug_specific_chromosome(tasks_json_path: str, chromosome_to_debug: List[int], total_gpus: int):
    """
    加载任务，并使用带调试信息的调度函数来运行一个特定的染色体。
    """
    print(f"--- 载入任务文件: {tasks_json_path} ---")
    try:
        with open(tasks_json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"✅ 成功加载 {len(tasks)} 个任务")
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return

    print(f"--- 调试目标染色体: {chromosome_to_debug} ---")
    print(f"--- GPU总数: {total_gpus} ---\n")

    # 调用带有详细调试信息的版本
    final_trace, final_makespan = generate_trace_from_chromosome_debug(
        tasks_input=tasks,
        chromosome=chromosome_to_debug,
        total_gpus=total_gpus
    )
    
    print("\n\n" + "="*50)
    print("🎉 调度完成 🎉")
    print("="*50)

    if not final_trace:
        print("❌ 调度器未返回有效轨迹。")
        return

    # 格式化并打印最终的Trace结果
    trace_df = pd.DataFrame(
        final_trace,
        columns=[
            "Task_ID", "Stage", "GPUs_Count", "Time",
            "Start", "End", "DatasetID", "GPU_List",
        ],
    ).sort_values(by=['Start', 'End']).reset_index(drop=True)

    print("\n--- 最终生成的调度轨迹 (按开始时间排序) ---")
    print(trace_df.to_string())
    
    print(f"\n\n🏆 最终计算出的 Makespan = {final_makespan:.2f}s")


if __name__ == '__main__':
    # --- 1. 请在这里配置您的信息 ---
    
    # 您的JSON文件路径
    TASKS_FILE_PATH = "generated_schedules/wan/schedule_4_tasks.json" 
    
    # 您要调试的特定染色体
    CHROMOSOME_TO_DEBUG = [4, 8, 10, 8, 4, 8, 4, 8]
    
    # 您的GPU总数
    TOTAL_GPUS = 16

    # --- 2. 运行调试器 ---
    debug_specific_chromosome(TASKS_FILE_PATH, CHROMOSOME_TO_DEBUG, TOTAL_GPUS)



#发现关键路径这种启发式会在调度GPU时有严格的顺序要求，