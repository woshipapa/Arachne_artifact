import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def solve_scheduling_pyomo(num_gpus, data_items, modules, proc_times_input, sp_choices_input):
    """
    使用 Pyomo 解决资源约束调度问题。

    Args:
        num_gpus (int): 可用的 GPU 总数 (N)。
        data_items (list): 数据项的列表, e.g., ['D_97', ...]。
        modules (list): 模块的列表, e.g., ['VAE', 'DiT']。
        proc_times_input (dict): 嵌套的字典，包含原始的处理时间。
        sp_choices_input (dict): 字典，包含每个任务可用的 SP 选项。

    Returns:
        tuple: 包含状态、最大完工时间、任务详情和所选 SP 模式的元组。
    """
    # --- 数据预处理 ---
    J_tasks = [(d, m) for d in data_items for m in modules]
    J_indices = list(range(len(J_tasks)))
    task_map = {task: idx for idx, task in enumerate(J_tasks)}
    inv_task_map = {idx: task for task, idx in task_map.items()}

    processing_times_data = {}
    sp_options_per_task = {}
    for (data_item, module), sp_options in sp_choices_input.items():
        task_idx = task_map[(data_item, module)]
        sp_options_per_task[task_idx] = sp_options
        for sp in sp_options:
            processing_times_data[task_idx, sp] = proc_times_input[data_item][module][sp]

    pred_map = {}
    for d in data_items:
        vae_task_idx = task_map[(d, 'VAE')]
        dit_task_idx = task_map[(d, 'DiT')]
        pred_map[dit_task_idx] = vae_task_idx

    # 1. 创建模型 (Create Model)
    # =================================
    model = pyo.ConcreteModel("Optimal_Scheduler")

    # 2. 集合与参数 (Sets and Parameters)
    # =================================
    model.J = pyo.Set(initialize=J_indices)
    model.E = pyo.Set(initialize=range(2 * len(J_tasks)))
    
    # 创建一个包含所有有效 (任务, SP选项) 对的集合
    model.ValidX = pyo.Set(initialize=processing_times_data.keys(), dimen=2)
    
    model.N = pyo.Param(initialize=num_gpus)
    model.M_big = pyo.Param(initialize=100000)
    model.p = pyo.Param(model.ValidX, initialize=processing_times_data)
    model.Pred = pyo.Param(model.J, initialize=pred_map)

    # 3. 决策变量 (Decision Variables)
    # ====================================
    model.x = pyo.Var(model.ValidX, domain=pyo.Binary)
    model.s = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.c = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.T = pyo.Var(model.E, domain=pyo.NonNegativeReals)
    model.z = pyo.Var(model.J, model.E, domain=pyo.Binary)
    model.y = pyo.Var(model.J, model.E, domain=pyo.Binary)
    model.C_max = pyo.Var(domain=pyo.NonNegativeReals)

    # 辅助变量
    model.u = pyo.Var(model.J, model.E, domain=pyo.Binary)
    # 创建一个包含所有有效 (任务, SP选项, 事件) 对的集合
    model.ValidW = pyo.Set(initialize=[(j, k, e) for j, k in model.ValidX for e in model.E], dimen=3)
    model.w = pyo.Var(model.ValidW, domain=pyo.Binary)

    # 4. 目标函数 (Objective Function)
    # =================================
    model.objective = pyo.Objective(expr=model.C_max, sense=pyo.minimize)

    # 5. 约束条件 (Constraints)
    # =============================
    # 1. 总完工时间定义
    def makespan_rule(m, j):
        return m.C_max >= m.c[j]
    model.makespan_constraint = pyo.Constraint(model.J, rule=makespan_rule)

    # 2. SP 模式选择唯一性
    def sp_choice_rule(m, j):
        return sum(m.x[j, k] for k in sp_options_per_task[j]) == 1
    model.sp_choice_constraint = pyo.Constraint(model.J, rule=sp_choice_rule)

    # 3. 完成时间计算
    def completion_time_rule(m, j):
        return m.c[j] == m.s[j] + sum(m.p[j, k] * m.x[j, k] for k in sp_options_per_task[j])
    model.completion_time_constraint = pyo.Constraint(model.J, rule=completion_time_rule)

    # 4. 前序约束
    def precedence_rule(m, j):
        if j in m.Pred:
            return m.s[j] >= m.c[m.Pred[j]]
        return pyo.Constraint.Skip
    model.precedence_constraint = pyo.Constraint(model.J, rule=precedence_rule)

    # 5. 事件结构与关联约束
    def event_ordering_rule(m, e):
        if e < max(m.E):
            return m.T[e] <= m.T[e+1]
        return pyo.Constraint.Skip
    model.event_ordering_constraint = pyo.Constraint(model.E, rule=event_ordering_rule)

    def start_event_assoc_rule(m, j):
        return sum(m.z[j, e] for e in m.E) == 1
    model.start_event_assoc_constraint = pyo.Constraint(model.J, rule=start_event_assoc_rule)

    def completion_event_assoc_rule(m, j):
        return sum(m.y[j, e] for e in m.E) == 1
    model.completion_event_assoc_constraint = pyo.Constraint(model.J, rule=completion_event_assoc_rule)

    def start_time_link_lower_rule(m, j, e):
        return m.s[j] >= m.T[e] - m.M_big * (1 - m.z[j, e])
    model.start_time_link_lower_constraint = pyo.Constraint(model.J, model.E, rule=start_time_link_lower_rule)
    
    def start_time_link_upper_rule(m, j, e):
        return m.s[j] <= m.T[e] + m.M_big * (1 - m.z[j, e])
    model.start_time_link_upper_constraint = pyo.Constraint(model.J, model.E, rule=start_time_link_upper_rule)

    def completion_time_link_lower_rule(m, j, e):
        return m.c[j] >= m.T[e] - m.M_big * (1 - m.y[j, e])
    model.completion_time_link_lower_constraint = pyo.Constraint(model.J, model.E, rule=completion_time_link_lower_rule)

    def completion_time_link_upper_rule(m, j, e):
        return m.c[j] <= m.T[e] + m.M_big * (1 - m.y[j, e])
    model.completion_time_link_upper_constraint = pyo.Constraint(model.J, model.E, rule=completion_time_link_upper_rule)

    # 6. 累积资源约束
    def active_status_rule(m, j, e):
        return m.u[j, e] == sum(m.z[j, e_prime] for e_prime in m.E if e_prime <= e) - \
                             sum(m.y[j, e_prime] for e_prime in m.E if e_prime <= e)
    model.active_status_constraint = pyo.Constraint(model.J, model.E, rule=active_status_rule)

    def linearization_w_le_x_rule(m, j, k, e):
        return m.w[j, k, e] <= m.x[j, k]
    model.linearization_w_le_x_constraint = pyo.Constraint(model.ValidW, rule=linearization_w_le_x_rule)

    def linearization_w_le_u_rule(m, j, k, e):
        return m.w[j, k, e] <= m.u[j, e]
    model.linearization_w_le_u_constraint = pyo.Constraint(model.ValidW, rule=linearization_w_le_u_rule)

    def linearization_w_ge_x_u_rule(m, j, k, e):
        return m.w[j, k, e] >= m.x[j, k] + m.u[j, e] - 1
    model.linearization_w_ge_x_u_constraint = pyo.Constraint(model.ValidW, rule=linearization_w_ge_x_u_rule)

    def resource_constraint_rule(m, e):
        return sum(k * m.w[j, k, e] for j, k in m.ValidX if (j, k, e) in m.ValidW) <= m.N
    model.resource_constraint = pyo.Constraint(model.E, rule=resource_constraint_rule)

    # 6. 求解模型 (Solve the Model)
    # =================================
    # 确保您已安装求解器, 例如 cbc 或 glpk
    # 'cbc' 是一个不错的开源选择
    solver = SolverFactory('cbc')
    results = solver.solve(model, tee=True) # tee=True 会显示求解器日志

    # 7. 提取并返回结果 (Extract and Return Results)
    # ==============================================
    status = results.solver.termination_condition
    if status == pyo.TerminationCondition.optimal:
        makespan = pyo.value(model.C_max)
        task_details = {}
        sp_modes = {}
        for j_idx in model.J:
            original_task = inv_task_map[j_idx]
            task_details[original_task] = {
                'start': pyo.value(model.s[j_idx]),
                'complete': pyo.value(model.c[j_idx])
            }
            for k in sp_options_per_task[j_idx]:
                if pyo.value(model.x[j_idx, k]) == 1:
                    sp_modes[original_task] = k
                    break
        return str(status), makespan, task_details, sp_modes
    else:
        return str(status), None, None, None


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

    print("--- 输入参数 (Pyomo) ---")
    print(f"GPU 总数 (N): {N_GPUS}")
    print(f"数据项: {data_items}")
    print(f"总任务数: {len(data_items) * len(modules)}")
    print("-" * 30 + "\n")

    # 调用求解器
    status, makespan, task_details, sp_modes = solve_scheduling_pyomo(
        num_gpus=N_GPUS,
        data_items=data_items,
        modules=modules,
        proc_times_input=proc_times_input,
        sp_choices_input=sp_choices_input
    )

    # 打印结果
    print(f"--- 求解结果 ---")
    print(f"求解状态: {status}")

    if status == 'optimal':
        print(f"\n最小总完工时间 (C_max): {makespan:.2f} 秒")
        print("\n--- 任务调度详情 ---")
        print(f"{'任务':<20} | {'SP模式':<8} | {'开始时间':<12} | {'完成时间':<12}")
        print("-" * 60)
        
        sorted_tasks = sorted(task_details.items(), key=lambda item: item[1]['start'])

        for task, times in sorted_tasks:
            sp = sp_modes[task]
            start_time = times['start']
            complete_time = times['complete']
            print(f"{str(task):<20} | {sp:<8} | {start_time:<12.2f} | {complete_time:<12.2f}")
    else:
        print("\n未能找到最优解。")
        print("可能的原因包括：问题不可行、求解时间不足或模型定义问题。")
        print("请检查求解器日志以获取更多信息。")

