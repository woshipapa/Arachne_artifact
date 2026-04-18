import collections
import random
import itertools
import math
import functools
from tqdm import tqdm # 引入tqdm来显示进度

# =============================================================================
# 适应度评估函数 (Fitness Evaluation) - 复用之前的核心调度器
# =============================================================================

# [复用上一版答案中的 get_critical_path_priority 和 _run_list_scheduler_v2 函数]
# ... 为保持代码简洁，此处省略，请确保你已经包含了这两个函数 ...
def get_critical_path_priority(task, predecessors, proc_times):
    if not hasattr(get_critical_path_priority, "cache"):
        get_critical_path_priority.cache = {}
    if task in get_critical_path_priority.cache:
        return get_critical_path_priority.cache[task]
    successors = [t for t, preds in predecessors.items() if task in preds]
    if not successors:
        return proc_times[task]
    max_succ_path = max(get_critical_path_priority(succ, predecessors, proc_times) for succ in successors)
    result = proc_times[task] + max_succ_path
    get_critical_path_priority.cache[task] = result
    return result

def _run_list_scheduler_v2(num_gpus, tasks, predecessors, fixed_sp_modes, fixed_proc_times, fixed_resources, priority_rule_func):
    total_num_tasks = len(tasks)
    schedule, completed_tasks, running_tasks, unscheduled_tasks = {}, set(), [], set(tasks)
    current_time, available_gpus = 0.0, num_gpus
    while len(completed_tasks) < total_num_tasks:
        ready_tasks = [t for t in unscheduled_tasks if not any(r[0] == t for r in running_tasks) and all(p in completed_tasks for p in predecessors.get(t, []))]
        ready_tasks.sort(key=priority_rule_func, reverse=True)
        for task_to_schedule in ready_tasks:
            res_needed = fixed_resources[task_to_schedule]
            if available_gpus >= res_needed:
                finish_time = current_time + fixed_proc_times[task_to_schedule]
                available_gpus -= res_needed
                running_tasks.append((task_to_schedule, finish_time))
                unscheduled_tasks.remove(task_to_schedule)
                schedule[task_to_schedule] = {'start': current_time, 'finish': finish_time, 'sp': fixed_sp_modes[task_to_schedule]}
        if not running_tasks:
            if unscheduled_tasks: return float('inf'), {}, {}
            break
        next_event_time = min(finish for task, finish in running_tasks)
        current_time = next_event_time
        for task, finish_time in list(running_tasks):
            if finish_time <= current_time:
                completed_tasks.add(task)
                running_tasks.remove((task, finish_time))
                available_gpus += fixed_resources[task]
    makespan = max(info['finish'] for info in schedule.values()) if schedule else 0
    task_details = {task: {'start': info['start'], 'finish': info['finish']} for task, info in schedule.items()}
    sp_modes = {task: info['sp'] for task, info in schedule.items()}
    return makespan, task_details, sp_modes

# =============================================================================
# 遗传算法 (Genetic Algorithm) 组件
# =============================================================================

def calculate_fitness(chromosome, tasks, num_gpus, predecessors, proc_times_input):
    """计算单个染色体（一个SP方案）的适应度（makespan）"""
    sp_config = {tasks[i]: chromosome[i] for i in range(len(tasks))}
    proc_times = {task: proc_times_input[task[0]][task[1]][sp] for task, sp in sp_config.items()}
    resources = {task: sp for task, sp in sp_config.items()}

    # 使用多种优先级规则评估，取最好的结果作为适应度
    best_makespan = float('inf')
    
    get_critical_path_priority.cache = {} # 清理缓存
    priority_rules = {
        "LPT": lambda t: proc_times[t],
        "SPT": lambda t: -proc_times[t],
        "CriticalPath": functools.partial(get_critical_path_priority, predecessors=predecessors, proc_times=proc_times)
    }

    for rule_func in priority_rules.values():
        makespan, _, _ = _run_list_scheduler_v2(num_gpus, tasks, predecessors, sp_config, proc_times, resources, rule_func)
        if makespan < best_makespan:
            best_makespan = makespan
    return best_makespan

def tournament_selection(population, fitnesses, k=3):
    """锦标赛选择法，随机选k个个体，选最好的那个"""
    best = None
    for _ in range(k):
        idx = random.randrange(len(population))
        if best is None or fitnesses[idx] < fitnesses[best]: # Makespan越小越好
            best = idx
    return population[best]

def crossover(parent1, parent2):
    """单点交叉"""
    if len(parent1) != len(parent2) or len(parent1) < 2:
        return parent1, parent2
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome, mutation_rate, sp_options_per_task):
    """变异：以一定概率随机改变某个基因"""
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            # 从该任务的可选SP值中重新随机选一个
            mutated_chromosome[i] = random.choice(sp_options_per_task[i])
    return mutated_chromosome

# =============================================================================
# 遗传算法主流程
# =============================================================================

def run_genetic_algorithm_scheduler(num_gpus, data_items, modules, proc_times_input, sp_choices_input,
                                    population_size=50, num_generations=100, mutation_rate=0.1, elitism_size=2):
    """
    使用遗传算法寻找最优SP配置
    """
    print("--- 运行遗传算法调度器 ---")
    
    # 1. 初始化
    tasks = [(d, m) for d in data_items for m in modules]
    predecessors = collections.defaultdict(list)
    for d in data_items:
        predecessors[(d, 'DiT')].append((d, 'VAE'))

    sp_options_per_task = []
    for task in tasks:
        valid_sps = [sp for sp in sp_choices_input[task] if sp <= num_gpus]
        if not valid_sps: raise ValueError(f"任务 {task} 没有任何可行的SP模式。")
        sp_options_per_task.append(valid_sps)

    # 2. 创建初始种群
    print(f"[步骤 1] 创建初始种群 (大小: {population_size})...")
    population = []
    for _ in range(population_size):
        chromosome = [random.choice(options) for options in sp_options_per_task]
        population.append(chromosome)

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    # 3. 进化循环
    for gen in tqdm(range(num_generations), desc="进化代数"):
        # 计算当前种群的适应度
        fitnesses = [calculate_fitness(chrom, tasks, num_gpus, predecessors, proc_times_input) for chrom in population]

        # 记录当代最佳
        min_fitness_idx = min(range(len(fitnesses)), key=fitnesses.__getitem__)
        if fitnesses[min_fitness_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[min_fitness_idx]
            best_overall_chromosome = population[min_fitness_idx]
            tqdm.write(f"代 {gen+1}: 发现新的最优解! Makespan: {best_overall_fitness:.2f}s")

        # 创建下一代种群
        new_population = []
        
        # 精英主义：直接保留最好的几个个体到下一代
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        for i in range(elitism_size):
            new_population.append(sorted_population[i])
            
        # 繁衍
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate, sp_options_per_task))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate, sp_options_per_task))
        
        population = new_population

    # 4. 返回找到的最优解
    print("\n[步骤 3] 进化完成。")
    # 用最优染色体，重新跑一次调度以获取详细结果
    final_sp_config = {tasks[i]: best_overall_chromosome[i] for i in range(len(tasks))}
    proc_times = {task: proc_times_input[task[0]][task[1]][sp] for task, sp in final_sp_config.items()}
    resources = {task: sp for task, sp in final_sp_config.items()}

    best_makespan = float('inf')
    best_details, best_modes = {}, {}

    get_critical_path_priority.cache = {}
    priority_rules = {
        "LPT": lambda t: proc_times[t], "SPT": lambda t: -proc_times[t],
        "CriticalPath": functools.partial(get_critical_path_priority, predecessors=predecessors, proc_times=proc_times)
    }

    for rule_func in priority_rules.values():
        makespan, details, modes = _run_list_scheduler_v2(num_gpus, tasks, predecessors, final_sp_config, proc_times, resources, rule_func)
        if makespan < best_makespan:
            best_makespan, best_details, best_modes = makespan, details, modes

    return best_makespan, best_details, best_modes

# =============================================================================
# 主程序入口 (使用遗传算法)
# =============================================================================
if __name__ == '__main__':
    # --- 输入数据保持不变 ---
    N_GPUS = 16
    data_items = ['D_101', 'D2_45', 'D_113', 'D1_113']
    modules = ['VAE', 'DiT']
    # [省略 proc_times_input 和 sp_choices_input 的定义，与上一版相同]
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

    # 调用遗传算法求解器
    makespan, task_details, sp_modes = run_genetic_algorithm_scheduler(
        num_gpus=N_GPUS,
        data_items=data_items,
        modules=modules,
        proc_times_input=proc_times_input,
        sp_choices_input=sp_choices_input,
        # --- GA参数，你可以调整这些值 ---
        population_size=200,      # 种群大小
        num_generations=500,     # 进化代数
        mutation_rate=0.1,       # 变异率
        elitism_size=2           # 精英个体数量
    )

    # --- 打印结果部分保持不变 ---
    print(f"\n--- 最终结果 ---")
    if makespan > 0 and makespan != float('inf'):
        print(f"\n遗传算法找到的最佳总完工时间 (Makespan): {makespan:.2f} 秒")
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

