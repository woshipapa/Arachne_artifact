import pulp
import numpy as np

def solve_scheduling_milp(num_gpus, J, J_indices, task_map, pred_map, processing_times, sp_options_per_task):
    """
    使用混合整数线性规划解决资源约束调度问题。

    Args:
        num_gpus (int): 可用的 GPU 总数 (N)。
        J (list): 任务元组的列表, e.g., [('D_97', 'VAE'), ...]。
        J_indices (list): 任务的数字索引列表。
        task_map (dict): 从任务元组到其数字索引的映射。
        pred_map (dict): 任务索引的前序关系映射。
        processing_times (dict): 一个字典，包含每个任务索引 j 和 SP 选项 k 的处理时间 p_j,k。
                                 键是元组 (j, k)，值是时间。
        sp_options_per_task (dict): 一个字典，将每个任务索引 j 映射到其可用的 SP 选项列表。

    Returns:
        tuple: 包含状态、最大完工时间、任务详情和所选 SP 模式的元组。
    """
    # 1. 模型构成 (Model Components)
    # =================================
    N = num_gpus
    num_tasks = len(J)
    E = list(range(2 * num_tasks))
    M_big = 100000  # 一个足够大的正数
    p = processing_times

    # 2. 初始化模型 (Initialize Model)
    # ====================================
    model = pulp.LpProblem("Resource_Constrained_Scheduling", pulp.LpMinimize)

    # 3. 决策变量 (Decision Variables)
    # ====================================
    # x_j,k: 定义在每个任务可用的 SP 选项上
    x_indices = [(j, k) for j in J_indices for k in sp_options_per_task[j]]
    x = pulp.LpVariable.dicts("x", x_indices, cat='Binary')
    
    s = pulp.LpVariable.dicts("s", J_indices, lowBound=0, cat='Continuous')
    c = pulp.LpVariable.dicts("c", J_indices, lowBound=0, cat='Continuous')
    T = pulp.LpVariable.dicts("T", E, lowBound=0, cat='Continuous')
    z = pulp.LpVariable.dicts("z", (J_indices, E), cat='Binary')
    y = pulp.LpVariable.dicts("y", (J_indices, E), cat='Binary')
    C_max = pulp.LpVariable("C_max", lowBound=0, cat='Continuous')

    # 辅助变量
    u = pulp.LpVariable.dicts("u", (J_indices, E), cat='Binary')
    w_indices = [(j, k, e) for j in J_indices for k in sp_options_per_task.get(j, []) for e in E]
    w = pulp.LpVariable.dicts("w", w_indices, cat='Binary')

    # 4. 目标函数 (Objective Function)
    # =================================
    model += C_max, "Minimize_Makespan"

    # 5. 约束条件 (Constraints)
    # =============================
    # 1. 总完工时间定义
    for j in J_indices:
        model += C_max >= c[j], f"Makespan_Constraint_{j}"

    # 2. SP 模式选择唯一性
    for j in J_indices:
        model += pulp.lpSum(x[j, k] for k in sp_options_per_task[j]) == 1, f"SP_Choice_Uniqueness_{j}"

    # 3. 完成时间计算
    for j in J_indices:
        model += c[j] == s[j] + pulp.lpSum(p[j, k] * x[j, k] for k in sp_options_per_task[j]), f"Completion_Time_{j}"

    # 4. 前序约束
    for j, pred_j in pred_map.items():
        model += s[j] >= c[pred_j], f"Precedence_{pred_j}_to_{j}"

    # 5. 事件结构与关联约束
    for e in range(len(E) - 1):
        model += T[e] <= T[e+1], f"Event_Time_Ordering_{e}"

    for j in J_indices:
        model += pulp.lpSum(z[j][e] for e in E) == 1, f"Start_Event_Association_{j}"
        model += pulp.lpSum(y[j][e] for e in E) == 1, f"Completion_Event_Association_{j}"

    for j in J_indices:
        for e in E:
            model += s[j] >= T[e] - M_big * (1 - z[j][e]), f"Start_Time_Link_Lower_{j}_{e}"
            model += s[j] <= T[e] + M_big * (1 - z[j][e]), f"Start_Time_Link_Upper_{j}_{e}"
            model += c[j] >= T[e] - M_big * (1 - y[j][e]), f"Completion_Time_Link_Lower_{j}_{e}"
            model += c[j] <= T[e] + M_big * (1 - y[j][e]), f"Completion_Time_Link_Upper_{j}_{e}"

    # 6. 累积资源约束
    for j in J_indices:
        for e in E:
            model += u[j][e] == pulp.lpSum(z[j][e_prime] for e_prime in E if e_prime <= e) - \
                                pulp.lpSum(y[j][e_prime] for e_prime in E if e_prime <= e), f"Active_Status_{j}_{e}"

    for j in J_indices:
        for k in sp_options_per_task[j]:
            for e in E:
                model += w[j, k, e] <= x[j, k], f"Linearization_w_le_x_{j}_{k}_{e}"
                model += w[j, k, e] <= u[j][e], f"Linearization_w_le_u_{j}_{k}_{e}"
                model += w[j, k, e] >= x[j, k] + u[j][e] - 1, f"Linearization_w_ge_x_u_{j}_{k}_{e}"
    
    for e in E:
        model += pulp.lpSum(k * w[j, k, e] for j in J_indices for k in sp_options_per_task[j]) <= N, f"Resource_Constraint_{e}"

    # 6. 求解模型 (Solve the Model)
    # =================================
    solver = pulp.PULP_CBC_CMD(msg=True) 
    model.solve(solver)

    # 7. 提取并返回结果 (Extract and Return Results)
    # ==============================================
    status = pulp.LpStatus[model.status]
    if status == 'Optimal':
        makespan = pulp.value(C_max)
        task_details = {}
        sp_modes = {}
        inv_task_map = {idx: task for task, idx in task_map.items()}
        for j_idx in J_indices:
            original_task = inv_task_map[j_idx]
            task_details[original_task] = {
                'start': pulp.value(s[j_idx]),
                'complete': pulp.value(c[j_idx])
            }
            for k in sp_options_per_task[j_idx]:
                if pulp.value(x[j_idx, k]) == 1:
                    sp_modes[original_task] = k
                    break
        return status, makespan, task_details, sp_modes
    else:
        return status, None, None, None


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

    # --- 数据预处理 ---
    # 创建任务列表和映射
    J = [(d, m) for d in data_items for m in modules]
    J_indices = list(range(len(J)))
    task_map = {task: idx for idx, task in enumerate(J)}

    # 为 PuLP 创建扁平化的处理时间字典
    processing_times_data = {}
    for (data_item, module), sp_options in sp_choices_input.items():
        task_idx = task_map[(data_item, module)]
        for sp in sp_options:
            processing_times_data[task_idx, sp] = proc_times_input[data_item][module][sp]

    # 创建每个任务索引的 SP 选项字典
    sp_options_per_task = {task_map[task]: options for task, options in sp_choices_input.items()}

    # 创建前序关系映射
    pred_map = {}
    for d in data_items:
        vae_task_idx = task_map[(d, 'VAE')]
        dit_task_idx = task_map[(d, 'DiT')]
        pred_map[dit_task_idx] = vae_task_idx
    
    print("--- 输入参数 ---")
    print(f"GPU 总数 (N): {N_GPUS}")
    print(f"数据项: {data_items}")
    print(f"总任务数 (|J|): {len(J)}")
    print("-" * 30 + "\n")

    # 调用求解器
    status, makespan, task_details, sp_modes = solve_scheduling_milp(
        num_gpus=N_GPUS,
        J=J,
        J_indices=J_indices,
        task_map=task_map,
        pred_map=pred_map,
        processing_times=processing_times_data,
        sp_options_per_task=sp_options_per_task
    )

    # 打印结果
    print(f"--- 求解结果 ---")
    print(f"求解状态: {status}")

    if status == 'Optimal':
        print(f"\n最小总完工时间 (C_max): {makespan:.2f} 秒")
        print("\n--- 任务调度详情 ---")
        print(f"{'任务':<20} | {'SP模式':<8} | {'开始时间':<12} | {'完成时间':<12}")
        print("-" * 60)
        
        # 按开始时间排序以方便查看
        sorted_tasks = sorted(task_details.items(), key=lambda item: item[1]['start'])

        for task, times in sorted_tasks:
            sp = sp_modes[task]
            start_time = times['start']
            complete_time = times['complete']
            print(f"{str(task):<20} | {sp:<8} | {start_time:<12.2f} | {complete_time:<12.2f}")
    else:
        print("\n未能找到最优解。")
        print("可能的原因包括：问题不可行、求解时间不足或模型定义问题。")

