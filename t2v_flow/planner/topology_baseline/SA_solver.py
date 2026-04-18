# t2v_flow/planner/topology_baseline_solver.py

import time
import math
import random
import copy

def solve_topology_baseline(
    tasks,
    n_gpus,
    topology_model,
    time_limit_sec=60, # 确保时间够长
    log=True,
    time_scale=1000,
    initial_temp=500.0, # 提高初始温度，增加初期探索性
    cooling_rate=0.999, # 减缓降温速度 (原来是0.995太快了)
    patience=2000,      # 新增参数：忍耐度。如果2000次迭代没更新最优解，就重启
):
    
    t0 = time.time()
    
    # -----------------------------------------------------
    # 1. Precompute Valid Subsets (保持不变)
    # -----------------------------------------------------
    from itertools import combinations
    valid_subsets_cache = {}
    needed_sps = set()
    for t in tasks:
        for sp, _ in t["vae"]: needed_sps.add(sp)
        for sp, _ in t["dit"]: needed_sps.add(sp)
        
    # print(f"[SA] Precomputing GPU subsets...", flush=True)
    for sp in needed_sps:
        if n_gpus > 16 and sp > 2:
             subs = set()
             while len(subs) < 1000: # 采样数稍微改小点提高速度
                 subs.add(tuple(sorted(random.sample(range(n_gpus), sp))))
             valid_subsets_cache[sp] = list(subs)
        else:
            valid_subsets_cache[sp] = list(combinations(range(n_gpus), sp))

    # -----------------------------------------------------
    # 2. Helper Functions (保持不变)
    # -----------------------------------------------------
    def generate_random_solution():
        sol = []
        for t in tasks:
            v_idx = random.randrange(len(t["vae"]))
            v_sp = t["vae"][v_idx][0]
            v_sub = random.choice(valid_subsets_cache[v_sp])
            
            d_idx = random.randrange(len(t["dit"]))
            d_sp = t["dit"][d_idx][0]
            d_sub = random.choice(valid_subsets_cache[d_sp])
            
            sol.append({
                "task_id": t["task_id"],
                "vae_idx": v_idx, "vae_subset": v_sub,
                "dit_idx": d_idx, "dit_subset": d_sub
            })
        order = list(range(len(tasks)))
        random.shuffle(order)
        return sol, order

    def evaluate(solution, execution_order):
        gpu_free_time = [0.0] * n_gpus
        max_makespan = 0
        detailed_schedule = []

        for tid_idx in execution_order:
            dec = solution[tid_idx]
            task_info = tasks[tid_idx]
            
            # --- VAE ---
            v_sp, v_base_dur = task_info["vae"][dec["vae_idx"]]
            v_subset = dec["vae_subset"]
            try: penalty, _ = topology_model.score_bitmask(v_subset)
            except: penalty = 1.0
            v_real_dur = v_base_dur * penalty
            
            v_start = max(gpu_free_time[g] for g in v_subset)
            v_end = v_start + v_real_dur
            for g in v_subset: gpu_free_time[g] = v_end
                
            # --- DiT ---
            d_sp, d_base_dur = task_info["dit"][dec["dit_idx"]]
            d_subset = dec["dit_subset"]
            try: penalty, _ = topology_model.score_bitmask(d_subset)
            except: penalty = 1.0
            d_real_dur = d_base_dur * penalty
            
            d_earliest = max(gpu_free_time[g] for g in d_subset)
            d_start = max(v_end, d_earliest)
            d_end = d_start + d_real_dur
            for g in d_subset: gpu_free_time[g] = d_end
            
            max_makespan = max(max_makespan, d_end)
            detailed_schedule.append({
                "task_id": task_info["task_id"],
                "vae": {"sp": v_sp, "start": v_start, "end": v_end, "subset": list(v_subset), "dur": v_real_dur},
                "dit": {"sp": d_sp, "start": d_start, "end": d_end, "subset": list(d_subset), "dur": d_real_dur}
            })
        return max_makespan, detailed_schedule

    # -----------------------------------------------------
    # 3. Main Loop: SA with RESTART Strategy
    # -----------------------------------------------------
    print(f">>> [SA-Topo] Starting Simulated Annealing (Limit: {time_limit_sec}s, Auto-Restart Enabled)...")
    
    # Initialize Global Best
    curr_sol, curr_order = generate_random_solution()
    curr_makespan, _ = evaluate(curr_sol, curr_order)
    
    best_sol = copy.deepcopy(curr_sol)
    best_detailed = [] # placeholder
    best_makespan = curr_makespan
    
    # Init history with the first random guess
    history = [{
        "t_sec": 0.0, 
        "makespan_ms": best_makespan * 1000, 
        "objective": best_makespan
    }]

    temperature = initial_temp
    
    iterations = 0
    iter_since_improvement = 0 # 计数器：多少次迭代没有变好了
    restarts = 0

    while True:
        now = time.time()
        if now - t0 > time_limit_sec:
            break
            
        iterations += 1
        
        # ================= Restart Logic =================
        # 如果卡住太久 (或者温度降太低)，就“重启炉子”
        if iter_since_improvement > patience or temperature < 0.001:
            restarts += 1
            # 策略：完全随机重启 (Random Restart)
            # 这能保证我们跳出当前的深坑，去探索完全不同的解空间
            curr_sol, curr_order = generate_random_solution()
            curr_makespan, _ = evaluate(curr_sol, curr_order)
            
            # 重置温度和计数器
            temperature = initial_temp
            iter_since_improvement = 0
            
            # (Optional) Log restart
            # if log: print(f"[SA] Restart #{restarts} at t={now-t0:.2f}s (stuck for {patience} iters)")
            
            continue # 跳过本次循环的剩余部分，直接开始新一轮
        # =================================================

        # --- Mutation (Same as before) ---
        new_sol = copy.deepcopy(curr_sol)
        new_order = copy.deepcopy(curr_order)
        mutation_type = random.random()
        
        if mutation_type < 0.4: # Change Subset
            t_idx = random.randrange(len(tasks))
            task = tasks[t_idx]
            if random.random() < 0.5:
                sp = task["vae"][new_sol[t_idx]["vae_idx"]][0]
                new_sol[t_idx]["vae_subset"] = random.choice(valid_subsets_cache[sp])
            else:
                sp = task["dit"][new_sol[t_idx]["dit_idx"]][0]
                new_sol[t_idx]["dit_subset"] = random.choice(valid_subsets_cache[sp])
        elif mutation_type < 0.7: # Change SP
            t_idx = random.randrange(len(tasks))
            task = tasks[t_idx]
            if random.random() < 0.5:
                new_idx = random.randrange(len(task["vae"]))
                new_sol[t_idx]["vae_idx"] = new_idx
                sp = task["vae"][new_idx][0]
                new_sol[t_idx]["vae_subset"] = random.choice(valid_subsets_cache[sp])
            else:
                new_idx = random.randrange(len(task["dit"]))
                new_sol[t_idx]["dit_idx"] = new_idx
                sp = task["dit"][new_idx][0]
                new_sol[t_idx]["dit_subset"] = random.choice(valid_subsets_cache[sp])
        else: # Swap Order
            i, j = random.sample(range(len(tasks)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
        # --- Evaluate ---
        new_makespan, new_detailed = evaluate(new_sol, new_order)
        
        # --- Acceptance ---
        delta = new_makespan - curr_makespan
        accept = False
        if delta < 0:
            accept = True
        else:
            if temperature > 0.001:
                # 注意：这里的scaling factor 10.0可能需要根据makespan的数量级调整
                # 如果makespan是秒级(如10.0)，delta可能是0.5，exp(-0.5*10/100) ~ 0.95 (Easy accept)
                prob = math.exp(-delta * 10.0 / temperature) 
                if random.random() < prob:
                    accept = True
        
        if accept:
            curr_sol = new_sol
            curr_order = new_order
            curr_makespan = new_makespan
            
            # --- Global Best Check ---
            if curr_makespan < best_makespan:
                best_makespan = curr_makespan
                best_sol = copy.deepcopy(curr_sol)
                best_detailed = new_detailed
                
                # 重置 stagnate 计数器，因为我们找到了更好的！
                iter_since_improvement = 0
                
                history.append({
                    "t_sec": time.time() - t0,
                    "makespan_ms": best_makespan * 1000,
                    "objective": best_makespan
                })
                if log:
                     print(f"[SA] New best: {best_makespan:.4f}s (iter {iterations}, temp {temperature:.1f}, restart {restarts})")
            else:
                # 接受了新解，但没有打破全局记录
                iter_since_improvement += 1
        else:
            # 没接受新解
            iter_since_improvement += 1
        
        # Cooling
        temperature *= cooling_rate

    # -----------------------------------------------------
    # 4. Final Output
    # -----------------------------------------------------
    if not best_detailed: # 防御性编程
        _, best_detailed = evaluate(best_sol, list(range(len(tasks))))

    return {
        "status": "FEASIBLE",
        "optimal_makespan": best_makespan,
        "schedule": best_detailed,
        "incumbents": history,
        "iterations": iterations,
        "restarts": restarts
    }