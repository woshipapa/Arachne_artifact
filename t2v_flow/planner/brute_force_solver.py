# t2v_flow/planner/brute_force_solver.py

import time
import math
import random
import copy

def solve_brute_force(
    tasks,
    n_gpus,
    time_limit_sec=30,
    log=False,
    time_scale=1000,
    method="random_sampling",  # 新增参数: 'dfs' or 'random_sampling'
):
    """
    Naive Baseline Solver (Slow Version).
    
    Strategy:
    - Pure Random Sampling (Monte Carlo): Randomly pick SP configs AND Randomly pick task order.
    - This creates a massive search space (Modes^N * N!), making convergence much slower
      and the curve more "natural" for a baseline comparison.
    """
    
    t0 = time.time()
    # random.seed(42) # 建议不要固定seed，或者每次换一个，保证足够的随机性
    
    # Pre-process decision options
    decision_points = []
    task_ids = []
    for task in tasks:
        tid = task["task_id"]
        task_ids.append(tid)
        decision_points.append({ "task_id": tid, "stage": "vae", "options": task["vae"] })
        decision_points.append({ "task_id": tid, "stage": "dit", "options": task["dit"] })
    
    n_decisions = len(decision_points)
    
    # Global state
    best_makespan = float('inf')
    best_schedule = []
    history = []
    iterations = 0

    # ==========================================
    # Greedy Scheduler (Inner Loop)
    # ==========================================
    def calculate_makespan_greedy(selected_configs, execution_order):
        """
        Args:
            selected_configs: dict, {tid: {'vae': (sp, dur), 'dit': ...}}
            execution_order: list of task_ids, determining which task to schedule first.
        """
        occupied_intervals = [] # (start, end, demand)
        task_schedules = {}
        current_makespan = 0

        # Helper: Find earliest start time (First Fit)
        def find_earliest_slot(duration, demand, min_start_time):
            candidates = {min_start_time}
            for _, end, _ in occupied_intervals:
                if end >= min_start_time:
                    candidates.add(end)
            
            sorted_candidates = sorted(list(candidates))
            
            for t_start in sorted_candidates:
                t_end = t_start + duration
                # Check collision
                is_valid = True
                
                # Check overlaps
                check_points = {t_start}
                for s, e, _ in occupied_intervals:
                    if s > t_start and s < t_end:
                        check_points.add(s)
                
                for pt in check_points:
                    usage = 0
                    for s, e, dem in occupied_intervals:
                        if s <= pt < e:
                            usage += dem
                    if usage + demand > n_gpus:
                        is_valid = False
                        break
                
                if is_valid:
                    return t_start
            return min_start_time

        # Schedule tasks based on the RANDOM execution_order
        for tid in execution_order:
            # 1. VAE
            vae_sp, vae_dur = selected_configs[tid]["vae"]
            vae_start = find_earliest_slot(vae_dur, vae_sp, 0)
            vae_end = vae_start + vae_dur
            occupied_intervals.append((vae_start, vae_end, vae_sp))
            
            # 2. DiT
            dit_sp, dit_dur = selected_configs[tid]["dit"]
            dit_start = find_earliest_slot(dit_dur, dit_sp, vae_end)
            dit_end = dit_start + dit_dur
            occupied_intervals.append((dit_start, dit_end, dit_sp))
            
            task_schedules[tid] = {
                "vae": {"start": vae_start, "end": vae_end, "sp": vae_sp, "dur": vae_dur},
                "dit": {"start": dit_start, "end": dit_end, "sp": dit_sp, "dur": dit_dur}
            }
            current_makespan = max(current_makespan, dit_end)
            
            # Pruning (Optional: Disable to make it slower/dumber)
            if current_makespan >= best_makespan:
                 break 
                
        return current_makespan, task_schedules

    # ==========================================
    # Main Loop: Random Sampling
    # ==========================================
    print(f">>> [Baseline] Starting Monte Carlo Random Search (Limit: {time_limit_sec}s)...")
    
    while True:
        # Check Time
        now = time.time()
        if now - t0 > time_limit_sec:
            break
        
        iterations += 1
        
        # 1. Randomly pick SP configurations
        current_configs = {}
        for task in tasks:
            tid = task["task_id"]
            # Random choice for VAE
            vae_opt = random.choice(task["vae"])
            # Random choice for DiT
            dit_opt = random.choice(task["dit"])
            
            current_configs[tid] = {
                "vae": (vae_opt[0], int(vae_opt[1] * time_scale)),
                "dit": (dit_opt[0], int(dit_opt[1] * time_scale))
            }
            
        # 2. Randomly shuffle execution order (CRITICAL for slowing down convergence)
        # Without this, sticking to id=0,1,2... is often too good of a heuristic.
        random_order = list(task_ids)
        random.shuffle(random_order)
        
        # 3. Evaluate
        makespan, schedule = calculate_makespan_greedy(current_configs, random_order)
        
        # 4. Update Best
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = schedule
            elapsed = time.time() - t0
            history.append({
                "t": elapsed,
                "makespan_ms": makespan,
                "objective": makespan / time_scale
            })
            if log:
                print(f"[BF-Random] New best: {makespan/time_scale:.4f}s (iter {iterations})")

    # ==========================================
    # Format Output
    # ==========================================
    final_schedule = []
    if best_schedule:
        for tid, stages in best_schedule.items():
            final_schedule.append({
                "task_id": tid,
                "vae": {k: v/time_scale if k in ['start','end','dur'] else v for k,v in stages['vae'].items()},
                "dit": {k: v/time_scale if k in ['start','end','dur'] else v for k,v in stages['dit'].items()},
            })

    return {
        "status": "FEASIBLE" if best_schedule else "INFEASIBLE",
        "optimal_makespan": best_makespan / time_scale if best_schedule else None,
        "schedule": final_schedule,
        "anytime": history,
        "iterations": iterations
    }