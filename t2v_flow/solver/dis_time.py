import pyomo.environ as pyo
import math

# =============================================================================
# 1. 可配置参数
# =============================================================================
N_gpus = 8 
M = 10000 
SOLVER_NAME = 'glpk'

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

proc_times = {i: {m: {k: math.ceil(t) for k, t in proc_times_float[i][m].items()} for m in modules} for i in data_items}
horizon = sum(max(proc_times[i][m].values()) if proc_times[i][m] else 0 for i in data_items for m in modules)
all_sps_in_data = set()
for i in data_items:
    for m in modules:
        all_sps_in_data.update(proc_times[i][m].keys())
sp_options = sorted(list(all_sps_in_data))
for i in data_items:
    for m in modules:
        for k in sp_options:
            if k not in proc_times[i][m]:
                proc_times[i][m][k] = M

# =============================================================================
# 3. 模型构建
# =============================================================================
model = pyo.ConcreteModel("Linearized_Time_Discretized_Scheduler")

model.I = pyo.Set(initialize=data_items)
model.M = pyo.Set(initialize=modules)
model.K = pyo.Set(initialize=sp_options)
model.T = pyo.RangeSet(0, horizon)

model.x = pyo.Var(model.I, model.M, model.K, within=pyo.Binary)
model.s = pyo.Var(model.I, model.M, within=pyo.NonNegativeIntegers, bounds=(0, horizon))
model.C_max = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, horizon))
model.u = pyo.Var(model.I, model.M, model.T, within=pyo.Binary, doc="1 if task (i,m) is active at time t")
# !! 新增变量 p: (i,m)在t时刻的实际GPU占用 !!
model.p = pyo.Var(model.I, model.M, model.T, within=pyo.NonNegativeReals, doc="GPU usage of task (i,m) at time t")

model.objective = pyo.Objective(expr=model.C_max, sense=pyo.minimize)

def proc_time_expr(model, i, m):
    return sum(model.x[i, m, k] * proc_times[i][m][k] for k in model.K)

def sp_val_expr(model, i, m):
    return sum(k * model.x[i, m, k] for k in model.K)

# --- 基础约束 (保持不变) ---
@model.Constraint(model.I, model.M)
def sp_assignment_rule(model, i, m):
    return sum(model.x[i, m, k] for k in model.K) == 1
@model.Constraint(model.I, model.M)
def makespan_rule(model, i, m):
    return model.C_max >= model.s[i, m] + proc_time_expr(model, i, m)
@model.Constraint(model.I)
def precedence_rule(model, i):
    return model.s[i, 'DiT'] >= model.s[i, 'VAE'] + proc_time_expr(model, i, 'VAE')

# --- 时间离散化逻辑约束 (保持不变) ---
@model.Constraint(model.I, model.M)
def active_duration_rule(model, i, m):
    return sum(model.u[i, m, t] for t in model.T) == proc_time_expr(model, i, m)

@model.Constraint(model.I, model.M, model.T)
def active_contiguity_rule(model, i, m, t):
    return model.s[i, m] <= t + M * (1 - model.u[i,m,t])

# --- !! 核心修正：线性化的资源约束 !! ---

# 1. 累积约束: 在任何时间点t，所有任务的实际占用p之和不能超N
@model.Constraint(model.T)
def cumulative_resource_rule(model, t):
    return sum(model.p[i, m, t] for i in model.I for m in model.M) <= N_gpus

# 2. 线性化约束: 定义 p 的行为，确保 p = sp * u
@model.Constraint(model.I, model.M, model.T)
def linearization_rule_1(model, i, m, t):
    # p的最大值不能超过被选中的SP值
    return model.p[i, m, t] <= sp_val_expr(model, i, m)

@model.Constraint(model.I, model.M, model.T)
def linearization_rule_2(model, i, m, t):
    # p的最大值不能超过 M * u (如果u=0, 则p=0)
    return model.p[i, m, t] <= M * model.u[i, m, t]

@model.Constraint(model.I, model.M, model.T)
def linearization_rule_3(model, i, m, t):
    # 如果u=1, 则强制 p >= sp
    return model.p[i, m, t] >= sp_val_expr(model, i, m) - M * (1 - model.u[i, m, t])

# =============================================================================
# 5. 求解与结果输出
# =============================================================================
solver = pyo.SolverFactory(SOLVER_NAME)
print(f"--- Starting Linearized Time-Discretized MIP (N_gpus = {N_gpus}, Horizon = {horizon}) ---")
print("WARNING: This model can be very slow to solve due to its large size.")
results = solver.solve(model, tee=True)
print("--- Optimization Finished ---")

# (结果输出部分代码和之前一样)
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print(f"\n✅ Optimal Solution Found!")
    print(f"   Optimal Makespan (C_max): {pyo.value(model.C_max):.2f} seconds")
    print("\n--- Optimal Schedule ---")
    tasks = []
    for i in model.I:
        for m in model.M:
            start_time = pyo.value(model.s[i, m])
            chosen_sp = 0
            for k in model.K:
                if pyo.value(model.x[i, m, k]) > 0.5:
                    chosen_sp = k
                    break
            duration = proc_times[i][m][chosen_sp]
            tasks.append({'name': f"({i}, {m})", 'start': start_time, 'end': start_time + duration, 'sp': chosen_sp})
    tasks.sort(key=lambda x: x['start'])
    for task in tasks:
        print(f"Task {task['name']:<12}: Use SP={task['sp']:<2} | Starts at {task['start']:>6.0f} | Ends at {task['end']:>6.0f}")
else:
    print(f"\n⚠️ Solver did not find an optimal solution. Status: {results.solver.termination_condition}")