import collections

def run_heuristic_scheduler(num_gpus, data_items, modules, proc_times_input, sp_choices_input):
    """
    使用基于优先规则的列表调度启发式算法解决资源约束调度问题。

    Args:
        num_gpus (int): 可用的 GPU 总数 (N)。
        data_items (list): 数据项的列表, e.g., ['D_97', ...]。
        modules (list): 模块的列表, e.g., ['VAE', 'DiT']。
        proc_times_input (dict): 嵌套的字典，包含原始的处理时间。
        sp_choices_input (dict): 字典，包含每个任务可用的 SP 选项。

    Returns:
        tuple: 包含总完工时间、任务详情和所选 SP 模式的元组。
    """
    print("--- 运行启发式调度算法 ---")
    
    # 1. 预处理和数据结构初始化
    # ==================================
    tasks = [(d, m) for d in data_items for m in modules]
    total_num_tasks = len(tasks)

    # 创建前序关系: DiT 任务依赖于同一数据项的 VAE 任务
    predecessors = collections.defaultdict(list)
    for d in data_items:
        vae_task = (d, 'VAE')
        dit_task = (d, 'DiT')
        predecessors[dit_task].append(vae_task)

    # 启发式SP模式选择: "速度优先"
    # 为每个任务选择能使其执行时间最短的SP模式 (前提是SP <= num_gpus)
    fixed_sp_modes = {}
    fixed_proc_times = {}
    fixed_resources = {}

    print("\n[步骤 1] 启发式SP模式选择 ('速度优先'):")
    for task in tasks:
        best_sp = -1
        min_time = float('inf')
        
        available_sps = sp_choices_input[task]
        for sp in available_sps:
            # 检查资源约束
            if sp <= num_gpus:
                time = proc_times_input[task[0]][task[1]][sp]
                if time < min_time:
                    min_time = time
                    best_sp = sp
        
        if best_sp == -1:
            raise ValueError(f"任务 {task} 没有任何可行的SP模式在 {num_gpus} GPU的限制下。")

        fixed_sp_modes[task] = best_sp
        fixed_proc_times[task] = min_time
        fixed_resources[task] = best_sp # 资源消耗等于SP度
        print(f"  - 任务 {task}: 选择 SP={best_sp} (耗时: {min_time}s, 占用GPU: {best_sp})")


    # 2. 调度算法初始化
    # ======================
    unscheduled_tasks = set(tasks)
    completed_tasks = set()
    running_tasks = []  # 存储元组 (task, finish_time)
    
    # 使用一个字典来存储最终的调度结果
    schedule = {}
    
    current_time = 0.0
    available_gpus = num_gpus
    
    print("\n[步骤 2] 开始事件驱动的列表调度...")

    # 3. 主调度循环
    # ======================
    while len(completed_tasks) < total_num_tasks:
        
        # 3.1 找出所有可以开始的任务 (Ready Tasks)
        # 一个任务是 "Ready" 的，如果它的所有前序任务都已完成
        ready_tasks = []
        for task in unscheduled_tasks:
            # 检查任务是否已经在运行
            if any(t == task for t, f_time in running_tasks):
                continue
            
            # 检查前序任务是否都已完成
            preds = predecessors.get(task, [])
            if all(p in completed_tasks for p in preds):
                ready_tasks.append(task)
        
        # 3.2 应用优先级规则: 最长处理时间优先 (LPT)
        ready_tasks.sort(key=lambda t: fixed_proc_times[t], reverse=True)
        
        # 3.3 在当前时间点，安排尽可能多的 Ready 任务
        for task_to_schedule in ready_tasks:
            res_needed = fixed_resources[task_to_schedule]
            if available_gpus >= res_needed:
                # 安排这个任务！
                start_time = current_time
                finish_time = current_time + fixed_proc_times[task_to_schedule]
                
                # 更新状态
                available_gpus -= res_needed
                running_tasks.append((task_to_schedule, finish_time))
                unscheduled_tasks.remove(task_to_schedule)
                
                # 记录调度结果
                schedule[task_to_schedule] = {
                    'start': start_time,
                    'finish': finish_time,
                    'sp': fixed_sp_modes[task_to_schedule]
                }
                print(f"  - t={current_time:.2f}s: 调度任务 {task_to_schedule} (完成于 {finish_time:.2f}s)")

        # 3.4 时间推进: 跳转到下一个事件点
        # 如果没有任务在运行，则调度结束
        if not running_tasks:
            if unscheduled_tasks:
                 print("\n错误: 出现死锁！没有正在运行的任务来推进时间，但仍有未调度的任务。")
                 print("这可能是因为剩余任务所需的资源 > 总资源。")
            break # 结束循环

        # 下一个事件点是所有正在运行的任务中最早的完成时间
        next_event_time = min(finish for task, finish in running_tasks)
        
        print(f"  - t={current_time:.2f}s: 无更多任务可调度。推进时间至下一个事件点: t={next_event_time:.2f}s")
        current_time = next_event_time
        
        # 3.5 处理在新的当前时间点完成的任务
        # 使用 list(running_tasks) 创建一个副本进行迭代，因为我们会在循环中修改它
        for task, finish_time in list(running_tasks):
            if finish_time == current_time:
                completed_tasks.add(task)
                running_tasks.remove((task, finish_time))
                # 释放资源
                released_gpus = fixed_resources[task]
                available_gpus += released_gpus
                print(f"  - t={current_time:.2f}s: 任务 {task} 完成，释放 {released_gpus} GPU。可用GPU: {available_gpus}")

    # 4. 计算最终结果并返回
    # ========================
    makespan = max(info['finish'] for info in schedule.values()) if schedule else 0
    
    # 为了清晰的输出，将任务详情和SP模式分开
    task_details = {task: {'start': info['start'], 'finish': info['finish']} for task, info in schedule.items()}
    sp_modes = {task: info['sp'] for task, info in schedule.items()}

    print("\n--- 启发式算法调度完成 ---")
    return makespan, task_details, sp_modes


if __name__ == '__main__':
    # =================================================
    # 输入参数 (Input Parameters) - 使用您的真实数据
    # =================================================
    N_GPUS = 8
    data_items = ['D_97', 'D_121', 'D_221', 'D_241']
    modules = ['VAE', 'DiT']

    proc_times_input = {
        'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94}},
        'D_121': {'VAE': {1: 52.76, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83}},
        'D_221': {'VAE': {2: 52.9, 4: 31.60, 8: 16.5}, 'DiT': {2: 74.8, 4: 40.35, 8: 18.36}},
        'D_241': {'VAE': {2: 63.77, 4: 30.01, 8: 17.29}, 'DiT': {2: 92.94, 4: 42.7, 8: 21.58}}
    }
    sp_choices_input = {
        ('D_97', 'VAE'): [1, 2, 4, 8], ('D_97', 'DiT'): [1, 2, 4, 8],
        ('D_121', 'VAE'): [1, 2, 4, 8], ('D_121', 'DiT'): [1, 2, 4, 8],
        ('D_221', 'VAE'): [2, 4, 8],    ('D_221', 'DiT'): [2, 4, 8],
        ('D_241', 'VAE'): [2, 4, 8],    ('D_241', 'DiT'): [2, 4, 8]
    }

    # 调用启发式求解器
    makespan, task_details, sp_modes = run_heuristic_scheduler(
        num_gpus=N_GPUS,
        data_items=data_items,
        modules=modules,
        proc_times_input=proc_times_input,
        sp_choices_input=sp_choices_input
    )

    # 打印结果
    print(f"\n--- 最终结果 ---")
    if makespan > 0:
        print(f"\n启发式解的总完工时间 (Makespan): {makespan:.2f} 秒")
        print("\n--- 任务调度详情 ---")
        print(f"{'任务':<20} | {'SP模式':<8} | {'开始时间':<12} | {'完成时间':<12}")
        print("-" * 60)
        
        sorted_tasks = sorted(task_details.items(), key=lambda item: item[1]['start'])

        for task, times in sorted_tasks:
            sp = sp_modes[task]
            start_time = times['start']
            complete_time = times['finish']
            print(f"{str(task):<20} | {sp:<8} | {start_time:<12.2f} | {complete_time:<12.2f}")
    else:
        print("\n未能生成有效的调度方案。")

