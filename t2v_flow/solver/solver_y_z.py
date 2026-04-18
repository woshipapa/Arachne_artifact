import pyomo.environ as pyo


# --- 1. Parameters ---

# Data, Modules, and SP options (Sets)
data_items = ['D_241', 'D_221', 'D_97', 'D_121']
modules = ['VAE', 'DiT']
# sp_options = [1, 2, 3, 4, 6, 8, 24]
sp_options = [1, 2, 4 , 8]

# A sufficiently large number for Big-M method
M = 10000 

N_gpus = 8
# 从数据中自动推断所有可用的SP选项
all_sps_in_data = set()

proc_times = {
    
    'D_97': {
        'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8},
        'DiT': {1: 29.84, 2: 18.3, 4: 13.21, 8: 18.17}
    },
    'D_121': {
        'VAE': {1: 47.60, 2: 28.04, 4: 15.7, 8: 10.03},
        "DiT": {1: 45.2, 2: 29.44, 4: 20.82, 8: 28.3}
    },
    'D_221': {
        'VAE': {2: 52.90, 4: 26.64, 8: 16.5},
        'DiT': {2: 76.8,  4: 45.48, 8: 43.6}
    },
    'D_241': {
        'VAE': {2: 55.47, 4: 30.01, 8: 17.29},
        'DiT': {2: 95.63, 4: 55.54, 8: 49.28}
    }


}

# 动态获取所有用到的SP选项
for i in data_items:
    for m in modules:
        all_sps_in_data.update(proc_times[i][m].keys())
sp_options = sorted(list(all_sps_in_data))



# 为缺失的SP选项填充一个极大的时间，使其不可能被选中
for i in data_items:
    for m in modules:
        for k in sp_options:
            if k not in proc_times[i][m]:
                proc_times[i][m][k] = M # 使用一个大数作为惩罚

model = pyo.ConcreteModel("Optimal_Scheduler")


# Define the sets on the model for easy indexing
model.I = pyo.Set(initialize=data_items)
model.M = pyo.Set(initialize=modules)
model.K = pyo.Set(initialize=sp_options)

task_pairs = [(i, m, j, n) for i in model.I for m in model.M 
              for j in model.I for n in model.M if (i, m) < (j, n)]
model.TaskPairs = pyo.Set(initialize=task_pairs, dimen=4)

model.x = pyo.Var(model.I, model.M, model.K, within=pyo.Binary)
model.s = pyo.Var(model.I, model.M, within=pyo.NonNegativeReals)
model.C_max = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(model.TaskPairs, within=pyo.Binary, doc="Sequencing order (0=A->B, 1=B->A)")
model.z = pyo.Var(model.TaskPairs, within=pyo.Binary, doc="Overlap permission (1=can overlap)")

model.objective = pyo.Objective(expr=model.C_max, sense=pyo.minimize)


# --- 3. Objective Function ---
# The goal is to minimize the makespan
model.objective = pyo.Objective(expr=model.C_max, sense=pyo.minimize)



def proc_time_expr(m, i, m_name):
    return sum(model.x[i, m_name, k] * proc_times[i][m_name][k] for k in model.K)

def sp_val_expr(model, i, m):
    return sum(k * model.x[i, m, k] for k in model.K)

# Constraint 1: SP Assignment
# For each task, exactly one SP must be chosen.
def sp_assignment_rule(model, i, m):
    return sum(model.x[i, m, k] for k in model.K) == 1
model.sp_assignment = pyo.Constraint(model.I, model.M, rule=sp_assignment_rule)

# Constraint 2: Makespan
# C_max must be greater than or equal to all task completion times.
def makespan_rule(model, i, m):
    return model.C_max >= model.s[i, m] + proc_time_expr(model, i, m)
model.makespan_con = pyo.Constraint(model.I, model.M, rule=makespan_rule)


# Constraint 3: Precedence
# DiT must start after VAE is finished.
def precedence_rule(model, i):
    return model.s[i, 'DiT'] >= model.s[i, 'VAE'] + proc_time_expr(model, i, 'VAE')
model.precedence = pyo.Constraint(model.I, rule=precedence_rule)

# model.resource_cons = pyo.ConstraintList()
# for i, m, j, n in model.TaskPairs:
#     # 遍历这对任务所有可能的SP选择组合 (k1 for task A, k2 for task B)
#     for k1 in sp_options:
#         for k2 in sp_options:
#             # 如果这个SP组合会超过GPU限制
#             if k1 + k2 > N_gpus:
#                 # 只有当优化器“同时”选择 x[i,m,k1]=1 和 x[j,n,k2]=1 时，
#                 # 我们才强制执行“无重叠”约束。
#                 # M * (2 - x_A - x_B) 是实现这个逻辑的“开关”。
                
#                 # 约束 A -> B (y=0)
#                 model.resource_cons.add(
#                     model.s[i,m] + proc_times[i][m][k1] <= model.s[j,n] 
#                     + M * model.y[i,m,j,n] 
#                     + M * (2 - model.x[i,m,k1] - model.x[j,n,k2])
#                 )
                
#                 # 约束 B -> A (y=1)
#                 model.resource_cons.add(
#                     model.s[j,n] + proc_times[j][n][k2] <= model.s[i,m] 
#                     + M * (1 - model.y[i,m,j,n]) 
#                     + M * (2 - model.x[i,m,k1] - model.x[j,n,k2])
#                 )


# Constraint 4: Resource Constraints (Non-overlap)
# For each pair of tasks (i,m) and (j,n)
# 对每一对不同的任务 (A, B) 应用以下三条规则
# A = (i,m), B = (j,n)

@model.Constraint(model.TaskPairs)
def resource_linking_rule(model, i, m, j, n):
    """规则1: 链接资源使用和“允许并行”开关z"""
    # 如果SP之和 > N, 则z必须为0 (不允许并行)
    # 如果SP之和 <= N, 则z可以为1 (允许并行)
    return sp_val_expr(model, i, m) + sp_val_expr(model, j, n) <= N_gpus + M * ( 1 - model.z[i, m, j, n])

@model.Constraint(model.TaskPairs)
def sequence_rule_1(model, i, m, j, n):
    """规则2: 决定顺序 (A -> B)"""
    # 如果必须串行(z=0) 且 B在A后(y=0), 则A必须在B开始前完成
    return model.s[i, m] + proc_time_expr(model, i, m) <= model.s[j, n] + M * model.y[i, m, j, n] + M * model.z[i, m, j, n]

@model.Constraint(model.TaskPairs)
def sequence_rule_2(model, i, m, j, n):
    """规则3: 决定顺序 (B -> A)"""
    # 如果必须串行(z=0) 且 A在B后(y=1), 则B必须在A开始前完成
    return model.s[j, n] + proc_time_expr(model, j, n) <= model.s[i, m] + M * (1 - model.y[i, m, j, n]) + M * model.z[i, m, j, n]



# --- 5. Solve the Model ---

# Specify the solver (e.g., 'glpk', 'gurobi', 'cplex')
solver = pyo.SolverFactory('glpk') 
# You need to have the Gurobi solver installed and licensed (or use an open-source one like 'glpk')

# Solve
print("--- Starting Optimization ---")
results = solver.solve(model, tee=True)
print("--- Optimization Finished ---")



# =============================================================================
# DEBUGGING SECTION: 检查 y 和 z 的值
# =============================================================================
print("\n--- Debugging Auxiliary Variables ---")
print("Checking task pairs that were scheduled in parallel but shouldn't have been...")

# 我们关注在 t=0 时刻启动的任务对
conflict_pairs_to_check = [
    ('D_241', 'DiT', 'D_221', 'DiT'), # 8+8=16 > 8, 应该冲突
    ('D_241', 'DiT', 'D_97', 'VAE'), # 8+4=12 > 8, 应该冲突
    ('D_221', 'DiT', 'D_121', 'VAE')  # 8+4=12 > 8, 应该冲突
]

for pair in conflict_pairs_to_check:
    # 确保我们查询的键是符合 (i,m) < (j,n) 顺序的
    i, m, j, n = pair
    if (i, m) > (j, n):
        i, m, j, n = j, n, i, m # 交换顺序
    
    # 获取z值（允许并行的开关）
    z_val = pyo.value(model.z[i, m, j, n])
    
    print(f"\nFor pair ({i},{m}) and ({j},{n}):")
    print(f"  - z value (permission to overlap) is: {z_val}")
    if z_val > 0.5:
        print("  - ANALYSIS: z=1 means the model BELIEVED these tasks could overlap, which is WRONG.")
    else:
        print("  - ANALYSIS: z=0 means the model KNEW they must be sequential.")
        # 如果z=0但它们仍然并行了，说明y变量的约束部分也失效了
        y_val = pyo.value(model.y[i, m, j, n])
        print(f"  - y value (sequencing order) is: {y_val}")
        print("  - ANALYSIS: If z=0, these tasks should have a strict order, but they both started at t=0. This indicates a deep model flaw.")



# --- 6. Display Results ---
# 检查求解结果并输出
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print(f"\n✅ Optimal Solution Found!")
    print(f"   Optimal Makespan (C_max): {pyo.value(model.C_max):.2f} seconds")
    print("\n--- Optimal Schedule ---")
    
    # 收集并排序任务以供显示
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
            tasks.append({
                'name': f"({i}, {m})", 
                'start': start_time, 
                'end': start_time + duration, 
                'sp': chosen_sp
            })
    
    tasks.sort(key=lambda x: x['start'])
    
    for task in tasks:
        print(f"Task {task['name']:<12}: Use SP={task['sp']:<2} | Starts at {task['start']:>6.2f} | Ends at {task['end']:>6.2f}")
        
elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("\n❌ Infeasible Solution.")
    print("   The problem, as defined, has no possible solution. Check constraints, especially N_gpus.")
else:
    print(f"\n⚠️ Solver did not find an optimal solution. Status: {results.solver.termination_condition}")
