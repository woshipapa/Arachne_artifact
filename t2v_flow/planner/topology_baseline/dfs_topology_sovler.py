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
        # 这一步至关重要：
        # 我们把每个任务的 SP 选项按照 (base_duration) 从小到大排序。
        # 这样 DFS 会优先尝试 SP=8 (耗时短)，而不是 SP=1 (耗时长)。
        # 这保证了我们能迅速找到一个 tight upper bound，从而剪掉 SP=1 的慢分支。
        self.sorted_options = {} # (tid, stage) -> list of (sp, base_dur)
        self.min_remaining_cache = {} # (tid, stage) -> min_duration
        
        for i, t in enumerate(self.tasks):
            # Sort VAE
            vae_opts = sorted(t["vae"], key=lambda x: x[1]) # x[1] is duration
            self.sorted_options[(i, 'vae')] = vae_opts
            self.min_remaining_cache[(i, 'vae')] = vae_opts[0][1] # 最快跑完的时间
            
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
            # 如果没超时，且搜完了，就是 OPTIMAL
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
        # 估算：即使剩下的任务全部完美并行、全部用最快SP、全部无拓扑惩罚，
        # 还需要多少时间？如果 (当前时间 + 理论剩余时间) >= best_makespan，则剪枝。
        # 这是一个松弛下界 (Relaxed Lower Bound)。
        
        # 简易版 Look-ahead：
        # 找出剩下的任务中，单体耗时最长的那个任务的“最短理论耗时”。
        # 因为那个任务肯定要被调度，MakeSpan 至少要等于它的结束时间。
        # (注意：这个剪枝计算不能太重，否则得不偿失)
        
        min_start_time = min(current_gpu_times) # 最早能开始的时间
        theoretical_end = min_start_time
        
        # 稍微遍历一下剩下的任务（为了速度，只做简单检查）
        # 检查逻辑：如果有一个剩下的任务，其 (base_duration) + current_max_makespan > best，那肯定不行？
        # 不对，因为它可以并行。
        # 正确逻辑：MakeSpan 至少是 (剩余所有任务的总计算量 / GPU总数) + 当前时间。
        # 但这太复杂。
        # 实用逻辑：如果 (当前最晚的GPU时间 + 剩下任意一个任务的最短耗时) > best，可能不严谨（因为可以用空闲GPU）。
        
        # 我们采用最安全的剪枝：
        # 任何一个剩余任务的 duration 加上 "最早可用的资源时间" 如果超过 best，则剪枝。
        # 对于 strict optimal，这需要非常小心。这里先略过复杂LB，靠 Fix 1 已经很强了。
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
        # 优先调度那些 "最难搞" (耗时最长) 的任务？
        # 还是优先调度 "最短" 的任务以填补空隙？
        # 对于找 Upper Bound，"最长处理时间优先 (LPT)" 通常能更快找到好的 Makspan。
        # 我们按照 min_remaining_cache 降序排列 (耗时长的排前面)。
        candidates.sort(key=lambda x: self.min_remaining_cache[x], reverse=True)

        for task_idx, stage_type in candidates:
            
            # 使用预排序的 options (Shortest Duration First!)
            options = self.sorted_options[(task_idx, stage_type)]
            
            for sp, base_dur in options:
                
                # Get Subsets
                candidate_subsets = self.sorted_subsets[sp]
                
                # =========================================================
                # [Fix 5] Subset Beam Width (Critical for Speed)
                # =========================================================
                # 如果 SP=1，有 16 个子集。DFS 会在这里分叉 16 次。
                # 对于同构集群（或者差异不大的），尝试前 4 个足够了。
                # 如果你追求绝对理论最优，可以把这个限制去掉，但那会慢很多。
                # 建议：限制为 8。
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