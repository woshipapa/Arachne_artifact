# t2v_flow/planner/optimal_fixed_sp.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple
from ortools.sat.python import cp_model

class IncumbentRecorder(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var: cp_model.IntVar):
        super().__init__()
        self.makespan_var = makespan_var
        self.t0 = time.time()
        self.history = []  # [{"t_sec":..., "makespan_ms":...}, ...]

    def on_solution_callback(self):
        self.history.append({
            "t_sec": time.time() - self.t0,
            "makespan_ms": int(self.Value(self.makespan_var)),
        })

def solve_fixed_sp_optimal_makespan(
    tasks: List[Dict[str, Any]],
    n_gpus: int,
    chromosome: List[int],          # [vae_sp0, dit_sp0, vae_sp1, dit_sp1, ...]
    time_limit_sec: float = 30.0,
    log: bool = False,
    record_anytime: bool = False,
) -> Dict[str, Any]:
    """
    精确：在固定 chromosome (SP选择) 的情况下，求最优调度 makespan。
    返回: {"status", "makespan_sec", "wall_time_sec", "anytime", "stats"}
    """
    model = cp_model.CpModel()

    # gene mapping
    gene_to_key: List[Tuple[int, str]] = []
    for t in tasks:
        tid = int(t["task_id"])
        gene_to_key.append((tid, "vae"))
        gene_to_key.append((tid, "dit"))

    assert len(chromosome) == len(gene_to_key)

    # build horizon
    horizon = 0
    dur_ms = {}
    sp_map = {}
    for gi, (tid, stage) in enumerate(gene_to_key):
        sp = int(chromosome[gi])
        table = dict(tasks[tid][stage]) if isinstance(tasks[tid], dict) and tid in range(len(tasks)) else None

        # 更稳：用 task_id 找对应 task（避免 task_id != index）
    id2task = {int(t["task_id"]): t for t in tasks}
    for gi, (tid, stage) in enumerate(gene_to_key):
        task = id2task[tid]
        sp = int(chromosome[gi])
        table = dict(task[stage])
        d = float(table[sp])
        dms = int(round(d * 1000.0))
        dur_ms[(tid, stage)] = dms
        sp_map[(tid, stage)] = sp
        horizon += dms

    # variables
    intervals = []
    demands = []

    start = {}
    end = {}
    itv = {}

    # one (mandatory) interval per stage (fixed duration and fixed demand)
    for t in tasks:
        tid = int(t["task_id"])
        for stage in ("vae", "dit"):
            s = model.NewIntVar(0, horizon, f"s_{stage}_{tid}")
            e = model.NewIntVar(0, horizon, f"e_{stage}_{tid}")
            d = dur_ms[(tid, stage)]
            interval = model.NewIntervalVar(s, d, e, f"itv_{stage}_{tid}")
            start[(tid, stage)] = s
            end[(tid, stage)] = e
            itv[(tid, stage)] = interval

            intervals.append(interval)
            demands.append(int(sp_map[(tid, stage)]))

        # precedence
        model.Add(start[(tid, "dit")] >= end[(tid, "vae")])

    # capacity
    model.AddCumulative(intervals, demands, n_gpus)

    # makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    for t in tasks:
        tid = int(t["task_id"])
        model.Add(makespan >= end[(tid, "dit")])

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = bool(log)
    solver.parameters.log_to_stdout = bool(log)
    # 让它更“慢”、更稳定（可选）
    # solver.parameters.cp_model_presolve = False
    # solver.parameters.cp_model_probing_level = 0

    t0 = time.time()
    rec = IncumbentRecorder(makespan) if record_anytime else None
    if record_anytime:
        st = solver.SolveWithSolutionCallback(model, rec)
    else:
        st = solver.Solve(model)
    wall = time.time() - t0

    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "makespan_sec": None,
            "wall_time_sec": wall,
            "anytime": rec.history if rec else [],
            "stats": {},
        }

    return {
        "status": "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE",
        "makespan_sec": solver.Value(makespan) / 1000.0,
        "wall_time_sec": wall,
        "anytime": rec.history if rec else [],
        "stats": {
            "num_conflicts": int(solver.NumConflicts()),
            "num_branches": int(solver.NumBranches()),
            "solver_wall_time": float(solver.WallTime()),
        }
    }
