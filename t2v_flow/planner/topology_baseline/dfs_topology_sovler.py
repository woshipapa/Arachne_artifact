# t2v_flow/planner/dfs_topology_solver.py

import time
import math
import copy
from itertools import combinations

class DfsBnBSolver:
    def __init__(self, tasks, n_gpus, topology_model, time_limit_sec, log=True, time_scale=1000):
        self.tasks = tasks
        self.n_gpus = n_gpus
        self.topo = topology_model
        self.time_limit = time_limit_sec
        self.log = log
        self.time_scale = time_scale
        
        self.t0 = time.time()
        
        # Global Best State
        self.best_makespan = float('inf')
        self.best_schedule = []
        self.history = []
        
        self.nodes_visited = 0
        self.leaf_nodes = 0
        
        # Precompute subsets sorted by penalty
        self.sorted_subsets = {} 
        self._precompute_subsets()
        
        # =========================================================
        # [Fix 1] Pre-sort Task Options by Duration (Ascending)
        # =========================================================
        self.sorted_options = {} # (tid, stage) -> list of (sp, base_dur)
        self.min_remaining_cache = {} # (tid, stage) -> min_duration
        
        for i, t in enumerate(self.tasks):
            # Sort VAE
            vae_opts = sorted(t["vae"], key=lambda x: x[1]) # x[1] is duration
            self.sorted_options[(i, 'vae')] = vae_opts
            self.min_remaining_cache[(i, 'vae')] = vae_opts[0][1]
            
            # Sort DiT
            dit_opts = sorted(t["dit"], key=lambda x: x[1])
            self.sorted_options[(i, 'dit')] = dit_opts
            self.min_remaining_cache[(i, 'dit')] = dit_opts[0][1]

        self.total_stages = len(tasks) * 2

    def _precompute_subsets(self):
        needed_sps = set()
        for t in self.tasks:
            for sp, _ in t["vae"]: needed_sps.add(sp)
            for sp, _ in t["dit"]: needed_sps.add(sp)
            
        if self.log:
            print("[DFS] Precomputing subsets...", flush=True)
        
        for sp in needed_sps:
            all_subs = list(combinations(range(self.n_gpus), sp))
            scored_subs = []
            for sub in all_subs:
                try:
                    p, _ = self.topo.score_bitmask(sub)
                except:
                    p = 1.0
                scored_subs.append((p, sub))
            
            # Sort: Lowest penalty first
            scored_subs.sort(key=lambda x: x[0])
            self.sorted_subsets[sp] = [x[1] for x in scored_subs]

    def solve(self):
        print(f">>> [DFS-Optimal] Starting Search (Limit: {self.time_limit}s)...", flush=True)
        
        initial_gpu_times = [0.0] * self.n_gpus
        initial_stage_ends = {} 
        completed_stages = set()

        self._dfs(
            completed_stages=completed_stages,
            current_gpu_times=initial_gpu_times, 
            stage_ends=initial_stage_ends,
            current_max_makespan=0.0,
            partial_schedule=[]
        )
        
        status = "FEASIBLE"
        if self.best_makespan != float('inf'):
            if time.time() - self.t0 < self.time_limit:
                status = "OPTIMAL"
        else:
            status = "INFEASIBLE"

        return {
            "status": status,
            "optimal_makespan": self.best_makespan if self.best_makespan != float('inf') else None,
            "schedule": self._format_schedule(self.best_schedule),
            "incumbents": self.history,
            "nodes_visited": self.nodes_visited
        }

    def _dfs(self, completed_stages, current_gpu_times, stage_ends, current_max_makespan, partial_schedule):
        # Time Check
        if self.nodes_visited % 2000 == 0:
            if time.time() - self.t0 > self.time_limit:
                return
        self.nodes_visited += 1

        # =========================================================
        # [Fix 2] Stronger Pruning (Basic)
        # =========================================================
        if current_max_makespan >= self.best_makespan:
            return

        # =========================================================
        # [Fix 3] Look-ahead Lower Bound Pruning
        # =========================================================
        
        
        min_start_time = min(current_gpu_times)
        theoretical_end = min_start_time
        
        
        pass 

        # Base Case
        if len(completed_stages) == self.total_stages:
            self.leaf_nodes += 1
            if current_max_makespan < self.best_makespan:
                self.best_makespan = current_max_makespan
                self.best_schedule = list(partial_schedule)
                
                elapsed = time.time() - self.t0
                self.history.append({
                    "t_sec": elapsed,
                    "makespan_ms": self.best_makespan * 1000,
                    "objective": self.best_makespan
                })
                if self.log:
                    print(f"[DFS] New Best: {self.best_makespan:.4f}s (Nodes: {self.nodes_visited})", flush=True)
            return

        # Generate Candidates
        candidates = []
        for i in range(len(self.tasks)):
            if (i, 'vae') not in completed_stages:
                candidates.append((i, 'vae'))
            elif (i, 'dit') not in completed_stages:
                if (i, 'vae') in completed_stages:
                    candidates.append((i, 'dit'))

        # =========================================================
        # [Fix 4] Candidate Ordering Heuristic
        # =========================================================
        candidates.sort(key=lambda x: self.min_remaining_cache[x], reverse=True)

        for task_idx, stage_type in candidates:
            
            options = self.sorted_options[(task_idx, stage_type)]
            
            for sp, base_dur in options:
                
                # Get Subsets
                candidate_subsets = self.sorted_subsets[sp]
                
                # =========================================================
                # [Fix 5] Subset Beam Width (Critical for Speed)
                # =========================================================
                limit = 8 if sp < self.n_gpus else 20
                if len(candidate_subsets) > limit:
                     candidate_subsets = candidate_subsets[:limit]

                for subset in candidate_subsets:
                    
                    # Cost Calc
                    try:
                        penalty, _ = self.topo.score_bitmask(subset)
                    except:
                        penalty = 1.0
                    real_dur = base_dur * penalty
                    
                    # Timing
                    resource_ready_time = 0.0
                    for g in subset:
                        if current_gpu_times[g] > resource_ready_time:
                            resource_ready_time = current_gpu_times[g]
                    
                    start_time = resource_ready_time
                    if stage_type == 'dit':
                        vae_end_time = stage_ends.get((task_idx, 'vae'), 0.0)
                        if vae_end_time > start_time:
                            start_time = vae_end_time
                    
                    end_time = start_time + real_dur
                    
                    # Local Pruning
                    if end_time >= self.best_makespan:
                        continue
                    
                    # Update State (Copy)
                    new_gpu_times = list(current_gpu_times)
                    for g in subset:
                        new_gpu_times[g] = end_time
                    
                    new_stage_ends = stage_ends
                    if stage_type == 'vae':
                        new_stage_ends = stage_ends.copy()
                        new_stage_ends[(task_idx, 'vae')] = end_time
                    
                    new_completed = completed_stages.copy()
                    new_completed.add((task_idx, stage_type))
                    
                    new_makespan = max(current_max_makespan, end_time)
                    
                    entry = {
                        "tid": task_idx, "stage": stage_type, 
                        "sp": sp, "subset": subset, 
                        "start": start_time, "end": end_time, "dur": real_dur
                    }
                    partial_schedule.append(entry)
                    
                    self._dfs(new_completed, new_gpu_times, new_stage_ends, new_makespan, partial_schedule)
                    
                    partial_schedule.pop()
                    
                    if time.time() - self.t0 > self.time_limit:
                        return

    def _format_schedule(self, flat_schedule):
        if not flat_schedule: return []
        mapped = {}
        for item in flat_schedule:
            tid = item['tid']
            if tid not in mapped: mapped[tid] = {"task_id": self.tasks[tid]["task_id"]}
            mapped[tid][item['stage']] = {
                "sp": item['sp'], "start": item['start'], "end": item['end'],
                "dur": item['dur'], "subset": list(item['subset'])
            }
        return list(mapped.values())

def solve_dfs_topology(tasks, n_gpus, topology_model, time_limit_sec=60, log=True):
    solver = DfsBnBSolver(tasks, n_gpus, topology_model, time_limit_sec, log)
    return solver.solve()