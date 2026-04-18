import collections
import random
import itertools
import math
import functools

# =============================================================================
# 升级版调度器 V2
# =============================================================================

def get_critical_path_priority(task, predecessors, proc_times):
    """
    计算任务在依赖图中的关键路径长度（从当前任务到最终完成）。
    这个值越高，说明这个任务后续的依赖链条越长，应该被优先处理。
    """
    # 使用缓存避免重复计算
    if not hasattr(get_critical_path_priority, "cache"):
        get_critical_path_priority.cache = {}
    if task in get_critical_path_priority.cache:
        return get_critical_path_priority.cache[task]

    # 找到所有依赖当前任务的后续任务
    successors = []
    for t, preds in predecessors.items():
        if task in preds:
            successors.append(t)
    
    # 如果没有后续任务，则其关键路径就是它自身的处理时间
    if not successors:
        return proc_times[task]

    # 否则，其关键路径是自身处理时间 + 最长的后续任务关键路径
    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times) for succ in successors)
    result = proc_times[task] + max_succ_path
    get_critical_path_priority.cache[task] = result
    return result


def _run_list_scheduler_v2(num_gpus, tasks, predecessors, fixed_sp_modes, fixed_proc_times, fixed_resources, priority_rule_func):
    """
    一个更通用的列表调度器，它接受一个优先级函数作为参数。
    """
    total_num_tasks = len(tasks)
    schedule = {}
    completed_tasks = set()
    running_tasks = []
    unscheduled_tasks = set(tasks)
    
    current_time = 0.0
    available_gpus = num_gpus
    
    while len(completed_tasks) < total_num_tasks:
        ready_tasks = []
        for task in unscheduled_tasks:
            if any(t == task for t, f_time in running_tasks):
                continue
            if all(p in completed_tasks for p in predecessors.get(task, [])):
                ready_tasks.append(task)
        
        # 动态应用传入的优先级规则
        ready_tasks.sort(key=priority_rule_func, reverse=True)
        
        # --- 后续逻辑与原版相同 ---
        for task_to_schedule in ready_tasks:
            res_needed = fixed_resources[task_to_schedule]
            if available_gpus >= res_needed:
                start_time = current_time
                finish_time = current_time + fixed_proc_times[task_to_schedule]
                available_gpus -= res_needed
                running_tasks.append((task_to_schedule, finish_time))
                unscheduled_tasks.remove(task_to_schedule)
                schedule[task_to_schedule] = {'start': start_time, 'finish': finish_time, 'sp': fixed_sp_modes[task_to_schedule]}

        if not running_tasks:
            if unscheduled_tasks: return float('inf'), {}, {}
            break

        next_event_time = min(finish for task, finish in running_tasks)
        current_time = next_event_time
        
        for task, finish_time in list(running_tasks):
            if finish_time <= current_time:
                completed_tasks.add(task)
                running_tasks.remove((task, finish_time))
                released_gpus = fixed_resources[task]
                available_gpus += released_gpus

    makespan = max(info['finish'] for info in schedule.values()) if schedule else 0
    task_details = {task: {'start': info['start'], 'finish': info['finish']} for task, info in schedule.items()}
    sp_modes = {task: info['sp'] for task, info in schedule.items()}
    
    return makespan, task_details, sp_modes


def run_smarter_heuristic_scheduler(num_gpus, data_items, modules, proc_times_input, sp_choices_input, max_exhaustive_search=50000, num_random_samples=2000):
    """
    使用多种优先级规则来评估每个SP配置，找到更好的解。
    """
    print("--- 运行增强版启发式算法 (多种优先级) ---")
    
    # --- 步骤 1 & 2 与原版相同 ---
    tasks = [(d, m) for d in data_items for m in modules]
    predecessors = collections.defaultdict(list)
    for d in data_items:
        predecessors[(d, 'DiT')].append((d, 'VAE'))

    sp_options_per_task = []
    for task in tasks:
        valid_sps = [sp for sp in sp_choices_input[task] if sp <= num_gpus]
        if not valid_sps: raise ValueError(f"任务 {task} 没有任何可行的SP模式。")
        sp_options_per_task.append(valid_sps)

    total_combinations = math.prod(len(opts) for opts in sp_options_per_task)
    print(f"\n[步骤 1] 发现总共有 {total_combinations} 种不同的SP模式组合。")

    # --- 步骤 3 与原版相同，生成待评估的SP配置列表 ---
    strategy_configs = []
    if total_combinations <= max_exhaustive_search:
        print(f"  - 组合数较小，将进行穷举搜索。")
        all_sp_combinations = itertools.product(*sp_options_per_task)
        for i, combo in enumerate(all_sp_combinations):
            strategy_configs.append((f'combo_{i+1}', {tasks[j]: combo[j] for j in range(len(tasks))}))
    else:
        print(f"  - 组合数过大，将采用采样策略。")
        strategy_configs.append(('max_speed', {tasks[i]: max(sp_options_per_task[i]) for i in range(len(tasks))}))
        strategy_configs.append(('min_resource', {tasks[i]: min(sp_options_per_task[i]) for i in range(len(tasks))}))
        for i in range(num_random_samples):
            strategy_configs.append((f'random_{i+1}', {tasks[j]: random.choice(sp_options_per_task[j]) for j in range(len(tasks))}))

    # 4. 循环评估所有策略，但对每个策略使用多种优先级规则
    best_makespan = float('inf')
    best_schedule_details = {}
    best_sp_modes = {}
    best_strategy_name = None
    best_priority_rule_name = None

    print(f"\n[步骤 2] 开始评估 {len(strategy_configs)} 种策略配置...")
    for i, (name, sp_config) in enumerate(strategy_configs):
        proc_times = {task: proc_times_input[task[0]][task[1]][sp] for task, sp in sp_config.items()}
        resources = {task: sp for task, sp in sp_config.items()}

        # 为当前SP配置定义优先级规则
        # 清除关键路径的缓存
        get_critical_path_priority.cache = {}
        priority_rules = {
            "LPT": lambda t: proc_times[t],  # 最长处理时间
            "SPT": lambda t: -proc_times[t], # 最短处理时间 (取负值实现反向排序)
            "CriticalPath": functools.partial(get_critical_path_priority, predecessors=predecessors, proc_times=proc_times)
        }

        # 对当前SP配置，尝试所有优先级规则
        for rule_name, rule_func in priority_rules.items():
            makespan, details, modes = _run_list_scheduler_v2(
                num_gpus, tasks, predecessors, sp_config, proc_times, resources, rule_func
            )
            
            if makespan < best_makespan:
                print(f"  >>> 新的最佳方案! 策略 '{name}' + 规则 '{rule_name}' 得到 Makespan: {makespan:.2f}s (旧最佳: {best_makespan:.2f}s)")
                best_makespan = makespan
                best_schedule_details = details
                best_sp_modes = modes
                best_strategy_name = name
                best_priority_rule_name = rule_name

    print(f"\n[步骤 3] 搜索完成。发现的最佳组合是 策略'{best_strategy_name}' + 优先级规则'{best_priority_rule_name}'。")
    return best_makespan, best_schedule_details, best_sp_modes

# =============================================================================
# 主程序入口 (使用新函数)
# =============================================================================
if __name__ == '__main__':
    # --- 输入数据保持不变 ---
    N_GPUS = 16
    data_items = ['D_101', 'D2_45', 'D_113', 'D1_113']
    modules = ['VAE', 'DiT']
    proc_times_input = {
        'D_101': {'VAE': {1: 7.387, 2: 3.87, 4: 2.27, 5: 1.95, 8: 1.48, 10: 1.17}, 'DiT': {4: 30.15, 5: 22.67, 8: 13.98, 10: 12.60}},
        'D2_45': {'VAE': {1: 5.71, 2: 2.96, 4: 1.77, 5: 1.51, 8: 1.13, 10: 0.88}, 'DiT': {4: 16.38, 5: 11.74, 8: 7.45, 10: 7.22}},
        'D_113': {'VAE': {1: 8.29, 2: 4.29, 4: 2.53, 5: 2.19, 8: 1.67, 10: 1.32}, 'DiT': {4: 35.69, 5: 28.47, 8: 17.26, 10: 14.96}},
        'D1_113': {'VAE': {1: 8.29, 2: 4.29, 4: 2.53, 5: 2.19, 8: 1.67, 10: 1.32}, 'DiT': {4: 35.69, 5: 28.47, 8: 17.26, 10: 14.96}}
    }
    sp_choices_input = {
        ('D_101', 'VAE'): [1, 2, 4, 5, 8, 10], ('D_101', 'DiT'): [4, 5, 8, 10],
        ('D2_45', 'VAE'): [1, 2, 4, 5, 8, 10], ('D2_45', 'DiT'): [4, 5, 8, 10],
        ('D_113', 'VAE'): [1, 2, 4, 5, 8, 10],   ('D_113', 'DiT'): [4, 5, 8, 10],
        ('D1_113', 'VAE'): [1, 2, 4, 5, 8, 10],   ('D1_113', 'DiT'): [4, 5, 8, 10]
    }
    
    # 调用增强版的启发式求解器
    makespan, task_details, sp_modes = run_smarter_heuristic_scheduler(
        num_gpus=N_GPUS,
        data_items=data_items,
        modules=modules,
        proc_times_input=proc_times_input,
        sp_choices_input=sp_choices_input,
        max_exhaustive_search=50000,
        num_random_samples=2000
    )

    # --- 打印结果部分保持不变 ---
    print(f"\n--- 最终结果 ---")
    if makespan > 0 and makespan != float('inf'):
        print(f"\n增强版启发式搜索找到的最佳总完工时间 (Makespan): {makespan:.2f} 秒")
        print("\n--- 任务调度详情 ---")
        print(f"{'任务':<25} | {'SP模式':<8} | {'开始时间':<12} | {'完成时间':<12}")
        print("-" * 65)
        
        sorted_tasks = sorted(task_details.items(), key=lambda item: item[1]['start'])

        for task, times in sorted_tasks:
            sp = sp_modes[task]
            start_time = times['start']
            complete_time = times['finish']
            task_name_str = f"('{task[0]}', '{task[1]}')"
            print(f"{task_name_str:<25} | {sp:<8} | {start_time:<12.2f} | {complete_time:<12.2f}")
    else:
        print("\n未能生成有效的调度方案。")