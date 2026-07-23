# t2v_flow/planner/optimal_solver.py
from __future__ import annotations

import math
import time
import csv
import ast
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model


# ============================================================
# Callback: record incumbents (anytime curve)
# ============================================================
class IncumbentRecorder(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var: cp_model.IntVar):
        super().__init__()
        self.makespan_var = makespan_var
        self.t0 = time.time()
        self.history: List[Dict[str, Any]] = []

    def on_solution_callback(self):
        self.history.append({
            "t_sec": time.time() - self.t0,
            "makespan_ms": int(self.Value(self.makespan_var)),
            "objective": float(self.ObjectiveValue()),
        })


# ============================================================
# Data structure
# ============================================================
@dataclass(frozen=True)
class PlacementOption:
    tid: int
    stage: str            # "vae" or "dit"
    sp: int
    subset: Tuple[int, ...]
    dur_ms: int
    dist_int: int


# ============================================================
# Utils
# ============================================================
def _ceil_mul_ms(base_s: float, penalty: float) -> int:
    base_ms = int(round(base_s * 1000.0))
    return int(math.ceil(base_ms * float(penalty)))


def load_ga_trace_csv(path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    Load GA schedule as CP-SAT warm start.
    """
    hint = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["Task_ID"])
            stage = row["Stage"].lower()
            sp = int(row["GPUs_Count"])
            subset = tuple(sorted(ast.literal_eval(row["GPU_List"])))
            start = float(row["Start"])
            end = float(row["End"])

            hint[(tid, stage)] = {
                "sp": sp,
                "subset": subset,
                "start_ms": int(round(start * 1000)),
                "end_ms": int(round(end * 1000)),
            }
    return hint


# ============================================================
# Main solver
# ============================================================
def solve_optimal_with_topology(
    tasks: List[Dict[str, Any]],
    n_gpus: int,
    topology_model,
    time_limit_sec: float = 30.0,
    log: bool = False,
    subset_cap_per_sp: Optional[int] = None,
    big_dist: int = 1_000_000,
    ga_trace_csv: Optional[str] = None,   # <<< NEW
) -> Dict[str, Any]:

    model = cp_model.CpModel()

    ga_hint = load_ga_trace_csv(ga_trace_csv) if ga_trace_csv else None

    # -----------------------------
    # Horizon
    # -----------------------------
    horizon = 0
    for t in tasks:
        horizon += max(int(sec * 1000) for _, sec in t["vae"])
        horizon += max(int(sec * 1000) for _, sec in t["dit"])
    horizon *= 2

    # -----------------------------
    # Generate placement options
    # -----------------------------
    placement_opts: List[PlacementOption] = []

    def gen_subsets(sp: int):
        all_sub = combinations(range(n_gpus), sp)
        if subset_cap_per_sp is None:
            return all_sub
        out = []
        for i, s in enumerate(all_sub):
            out.append(s)
            if i + 1 >= subset_cap_per_sp:
                break
        return out

    for task in tasks:
        tid = int(task["task_id"])
        for stage in ("vae", "dit"):
            for sp, base_s in task[stage]:
                sp = int(sp)

                forced_subsets = []
                if ga_hint and (tid, stage) in ga_hint:
                    if ga_hint[(tid, stage)]["sp"] == sp:
                        forced_subsets.append(ga_hint[(tid, stage)]["subset"])

                subsets = set(gen_subsets(sp)) | set(forced_subsets)

                for subset in subsets:
                    try:
                        penalty, dist = topology_model.score_bitmask(subset)
                    except Exception:
                        penalty, dist = 1.0, 0.0

                    dur_ms = _ceil_mul_ms(base_s, penalty)
                    dist_int = big_dist if dist is None or math.isinf(dist) else int(round(dist))

                    placement_opts.append(
                        PlacementOption(tid, stage, sp, tuple(subset), dur_ms, dist_int)
                    )

    # -----------------------------
    # Group by task-stage
    # -----------------------------
    opts_by_ts: Dict[Tuple[int, str], List[PlacementOption]] = {}
    for opt in placement_opts:
        opts_by_ts.setdefault((opt.tid, opt.stage), []).append(opt)

    # -----------------------------
    # Variables
    # -----------------------------
    var_map = {}
    gpu_intervals = [[] for _ in range(n_gpus)]

    for (tid, stage), opt_list in opts_by_ts.items():
        pres = []
        for opt in opt_list:
            s = model.NewIntVar(0, horizon, f"s_{tid}_{stage}_{opt.sp}_{opt.subset}")
            e = model.NewIntVar(0, horizon, f"e_{tid}_{stage}_{opt.sp}_{opt.subset}")
            p = model.NewBoolVar(f"use_{tid}_{stage}_{opt.sp}_{opt.subset}")

            itv = model.NewOptionalIntervalVar(s, opt.dur_ms, e, p, "itv")
            var_map[(tid, stage, opt.sp, opt.subset)] = (s, e, p)

            pres.append(p)
            for g in opt.subset:
                gpu_intervals[g].append(itv)

        model.AddExactlyOne(pres)

    # -----------------------------
    # Resource + precedence
    # -----------------------------
    for g in range(n_gpus):
        if gpu_intervals[g]:
            model.AddNoOverlap(gpu_intervals[g])

    for t in tasks:
        tid = int(t["task_id"])
        for v in opts_by_ts[(tid, "vae")]:
            sv, ev, pv = var_map[(tid, "vae", v.sp, v.subset)]
            for d in opts_by_ts[(tid, "dit")]:
                sd, _, pd = var_map[(tid, "dit", d.sp, d.subset)]
                model.Add(sd >= ev).OnlyEnforceIf([pv, pd])

    # -----------------------------
    # Makespan
    # -----------------------------
    makespan = model.NewIntVar(0, horizon, "makespan")
    for t in tasks:
        tid = int(t["task_id"])
        for d in opts_by_ts[(tid, "dit")]:
            _, e, p = var_map[(tid, "dit", d.sp, d.subset)]
            model.Add(makespan >= e).OnlyEnforceIf(p)

    model.Minimize(makespan)
    if ga_hint:
        ga_makespan_ms = max(h["end_ms"] for h in ga_hint.values())
        print(f"[CP] Injecting GA upper bound: makespan <= {ga_makespan_ms} ms")
        model.Add(makespan <= ga_makespan_ms)

    # -----------------------------
    # >>> GA Warm Start <<<
    # -----------------------------
    if ga_hint:
        print("[CP] Applying GA warm start hints...", flush=True)
        for (tid, stage), h in ga_hint.items():
            key = (tid, stage, h["sp"], h["subset"])
            if key in var_map:
                s, e, p = var_map[key]
                model.AddHint(p, 1)
                model.AddHint(s, h["start_ms"])
                model.AddHint(e, h["end_ms"])

    # -----------------------------
    # Solve
    # -----------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = log
    solver.parameters.log_to_stdout = log

    rec = IncumbentRecorder(makespan)
    status = solver.SolveWithSolutionCallback(model, rec)

    return {
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        "optimal_makespan": solver.Value(makespan) / 1000.0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "incumbents": rec.history,
        "solver_stats": {
            "wall_time_sec": solver.WallTime(),
            "num_branches": solver.NumBranches(),
            "num_conflicts": solver.NumConflicts(),
        }
    }
