# =============================================================================
# 诊断脚本 (DEBUGGING SCRIPT)
# =============================================================================
from ortools.sat.python import cp_model
import math
import platform
import sys

# --- 请在这里填写您的环境信息 ---
# 例如: OS = "Ubuntu 22.04"
OS_INFO = f"{platform.system()} {platform.release()}" 
PYTHON_VERSION = sys.version
# 请在终端运行 pip show ortools 后填写版本号, 例如 "9.9.3963"
ORTOOLS_VERSION = "PLEASE_FILL_IN" 

print("--- Environment Information ---")
print(f"OS: {OS_INFO}")
print(f"Python Version: {PYTHON_VERSION}")
print(f"OR-Tools Version: {ORTOOLS_VERSION}")
print("---------------------------------")


# =============================================================================
# 1. 可配置参数
# =============================================================================
N_gpus = 8 

# =============================================================================
# 2. 数据定义
# =============================================================================
# !! 重要：我们先从最小的问题规模开始测试 !!
data_items = ['D_97'] 
# data_items = ['D_97', 'D_121'] # 如果上面成功，再尝试这个
# data_items = ['D_97', 'D_121', 'D_221', 'D_241'] # 最后再尝试完整版

modules = ['VAE', 'DiT']
proc_times_float = {
    'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}},
    'D_121': {'VAE': {1: 47.60, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}},
    'D_221': {'VAE': {2: 52.90, 4: 26.64, 8: 16.5}, 'DiT': {2: 76.8, 4: 45.48, 8: 43.6}},
    'D_241': {'VAE': {2: 55.47, 4: 30.01, 8: 17.29}, 'DiT': {2: 95.63, 4: 55.54, 8: 49.28}}
}
horizon = int(sum(max(proc_times_float[i][m].values()) for i in data_items for m in modules) * 1.5 * 100)

# =============================================================================
# 3. CP-SAT 模型构建
# =============================================================================
model = cp_model.CpModel()
all_tasks = {}
all_intervals_with_demands = []

print("\n--- Starting Model Construction ---")

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

# 任务顺序约束
for i in data_items:
    for vae_option in all_tasks[(i, 'VAE')]:
        for dit_option in all_tasks[(i, 'DiT')]:
            model.Add(vae_option['interval'].EndExpr() <= dit_option['interval'].StartExpr()).OnlyEnforceIf(
                [vae_option['is_active'], dit_option['is_active']]
            )

# 资源约束
model.AddCumulative(N_gpus, all_intervals_with_demands)

# 目标函数
C_max = model.NewIntVar(0, horizon, 'makespan')
all_end_times = []
for task_key in all_tasks:
    for option in all_tasks[task_key]:
        all_end_times.append(option['interval'].EndExpr())
model.AddMaxEquality(C_max, all_end_times)
model.Minimize(C_max)

print("--- Model Construction Complete ---")

# =============================================================================
# 4. 求解与结果输出
# =============================================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0

print("\n--- Calling Solver... ---")
status = solver.Solve(model)
# 如果程序能打印下面这行，说明崩溃没有发生在求解过程中
print("--- Solver Call Finished. ---") 

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    # (结果输出部分代码和之前一样)
    print(f"\n✅ Solution Found!")
    print(f"   Optimal Makespan (C_max): {solver.ObjectiveValue() / 100.0:.2f} seconds")
    # ... etc ...
else:
    status_map = {cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "MODEL_INVALID", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE", cp_model.OPTIMAL: "OPTIMAL"}
    print(f"\n⚠️ No solution found. Status: {status_map.get(status, 'OTHER')}")