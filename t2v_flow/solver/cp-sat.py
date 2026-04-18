from ortools.sat.python import cp_model
import math

# =============================================================================
# 1. 可配置参数
# =============================================================================
N_gpus = 8 

# =============================================================================
# 2. 数据定义
# =============================================================================
data_items = ['D_97', 'D_121', 'D_221', 'D_241'] 
modules = ['VAE', 'DiT']
proc_times_float = {
    'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}},
    'D_121': {'VAE': {1: 47.60, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}},
    'D_221': {'VAE': {2: 52.90, 4: 26.64, 8: 16.5}, 'DiT': {2: 76.8, 4: 45.48, 8: 43.6}},
    'D_241': {'VAE': {2: 55.47, 4: 30.01, 8: 17.29}, 'DiT': {2: 95.63, 4: 55.54, 8: 49.28}}
}
horizon = sum(int(max(proc_times_float[i][m].values())*100) for i in data_items for m in modules)

# =============================================================================
# 3. CP-SAT 模型构建
# =============================================================================
model = cp_model.CpModel()
all_tasks = {}
all_intervals_with_demands = []

for i in data_items:
    for m in modules:
        task_sp_options = []
        possible_sps = proc_times_float[i][m].keys()
        for sp in possible_sps:
            duration = math.ceil(proc_times_float[i][m][sp] * 100)
            is_active = model.NewBoolVar(f'{i}_{m}_sp{sp}_active')
            start_var = model.NewIntVar(0, horizon, f'{i}_{m}_sp{sp}_start')
            end_var = model.NewIntVar(0, horizon, f'{i}_{m}_sp{sp}_end')
            interval_var = model.NewOptionalIntervalVar(start_var, duration, end_var, is_active, f'{i}_{m}_sp{sp}_interval')
            task_sp_options.append({'is_active': is_active, 'interval': interval_var, 'sp': sp})
            all_intervals_with_demands.append((interval_var, sp))
        all_tasks[(i, m)] = task_sp_options
        model.AddExactlyOne(option['is_active'] for option in task_sp_options)

# --- 添加约束 ---

# 1. 任务顺序约束 (VAE -> DiT) - !! 使用更稳健的写法 !!
for i in data_items:
    # 遍历VAE任务的每一个SP选项
    for vae_option in all_tasks[(i, 'VAE')]:
        # 遍历DiT任务的每一个SP选项
        for dit_option in all_tasks[(i, 'DiT')]:
            # 我们添加一条逻辑规则：
            # 如果 VAE的这个选项被激活 AND DiT的这个选项也被激活
            # 那么 VAE的结束时间必须 <= DiT的开始时间
            model.Add(vae_option['interval'].EndExpr() <= dit_option['interval'].StartExpr()).OnlyEnforceIf(
                [vae_option['is_active'], dit_option['is_active']]
            )

# 2. 资源约束 (AddCumulative) - 保持不变
model.AddCumulative(N_gpus, all_intervals_with_demands)

# --- 定义目标函数 ---
C_max = model.NewIntVar(0, horizon, 'makespan')
all_end_times = []
for task_key in all_tasks:
    for option in all_tasks[task_key]:
        all_end_times.append(option['interval'].EndExpr())
model.AddMaxEquality(C_max, all_end_times)
model.Minimize(C_max)

# =============================================================================
# 4. 求解与结果输出
# =============================================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"✅ Solution Found!")
    print(f"   Optimal Makespan (C_max): {solver.ObjectiveValue() / 100.0:.2f} seconds")
    print("\n--- Optimal Schedule ---")
    tasks = []
    for i in data_items:
        for m in modules:
            for option in all_tasks[(i, m)]:
                if solver.Value(option['is_active']):
                    start_time = solver.Value(option['interval'].StartExpr()) / 100.0
                    end_time = solver.Value(option['interval'].EndExpr()) / 100.0
                    tasks.append({'name': f"({i}, {m})", 'start': start_time, 'end': end_time, 'sp': option['sp']})
                    break
    tasks.sort(key=lambda x: x['start'])
    for task in tasks:
        print(f"Task {task['name']:<12}: Use SP={option['sp']:<2} | Starts at {task['start']:>6.2f} | Ends at {task['end']:>6.2f}")
else:
    status_map = {cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "MODEL_INVALID", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE", cp_model.OPTIMAL: "OPTIMAL"}
    print(f"\n⚠️ No solution found. Status: {status_map.get(status, 'OTHER')}")