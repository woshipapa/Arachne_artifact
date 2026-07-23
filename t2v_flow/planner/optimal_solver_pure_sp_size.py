# t2v_flow/planner/optimal_solver.py
from ortools.sat.python import cp_model
import time
class IncumbentRecorder(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var):
        super().__init__()
        self.makespan_var = makespan_var
        self.t0 = time.time()
        self.history = []  # list of dicts

    def on_solution_callback(self):
        self.history.append({
            "t": time.time() - self.t0,
            "makespan_ms": self.Value(self.makespan_var),
            "objective": self.ObjectiveValue(),
        })


def solve_optimal(
    tasks,
    n_gpus,
    time_limit_sec=30,
    log=False,
    record_anytime=True,
):
    model = cp_model.CpModel()

    # =========================
    # =========================
    horizon = 0
    for task in tasks:
        max_vae = max(int(t * 1000) for _, t in task["vae"])
        max_dit = max(int(t * 1000) for _, t in task["dit"])
        horizon += max_vae + max_dit

    intervals, demands = [], []
    vae_vars, dit_vars = {}, {}

    for task in tasks:
        tid = task["task_id"]

        vae_pres = []
        for sp, t in task["vae"]:
            dur = int(t * 1000)
            s = model.NewIntVar(0, horizon, f"s_vae_{tid}_{sp}")
            e = model.NewIntVar(0, horizon, f"e_vae_{tid}_{sp}")
            p = model.NewBoolVar(f"use_vae_{tid}_{sp}")
            itv = model.NewOptionalIntervalVar(s, dur, e, p, f"i_vae_{tid}_{sp}")
            vae_vars[(tid, sp)] = (s, e, p, dur)
            vae_pres.append(p)
            intervals.append(itv)
            demands.append(sp)
        model.AddExactlyOne(vae_pres)

        dit_pres = []
        for sp, t in task["dit"]:
            dur = int(t * 1000)
            s = model.NewIntVar(0, horizon, f"s_dit_{tid}_{sp}")
            e = model.NewIntVar(0, horizon, f"e_dit_{tid}_{sp}")
            p = model.NewBoolVar(f"use_dit_{tid}_{sp}")
            itv = model.NewOptionalIntervalVar(s, dur, e, p, f"i_dit_{tid}_{sp}")
            dit_vars[(tid, sp)] = (s, e, p, dur)
            dit_pres.append(p)
            intervals.append(itv)
            demands.append(sp)
        model.AddExactlyOne(dit_pres)

        for spv, _ in task["vae"]:
            sv, ev, pv, _ = vae_vars[(tid, spv)]
            for spd, _ in task["dit"]:
                sd, _, pd, _ = dit_vars[(tid, spd)]
                model.Add(sd >= ev).OnlyEnforceIf([pv, pd])

    model.AddCumulative(intervals, demands, n_gpus)

    makespan = model.NewIntVar(0, horizon, "makespan")
    for task in tasks:
        tid = task["task_id"]
        for sp, _ in task["dit"]:
            _, e, p, _ = dit_vars[(tid, sp)]
            model.Add(makespan >= e).OnlyEnforceIf(p)

    model.Minimize(makespan)

    # =========================
    # Solver + callback
    # =========================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.log_search_progress = bool(log)
    solver.parameters.log_to_stdout = bool(log)

    rec = IncumbentRecorder(makespan) if record_anytime else None

    if record_anytime:
        status = solver.SolveWithSolutionCallback(model, rec)
    else:
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "optimal_makespan": None,
            "schedule": [],
            "anytime": rec.history if rec else [],
        }

    # =========================
    # =========================
    def pick_stage(tid, stage_vars):
        for (t, sp), (s, e, p, dur) in stage_vars.items():
            if t == tid and solver.Value(p):
                return {
                    "sp": sp,
                    "start": solver.Value(s) / 1000,
                    "end": solver.Value(e) / 1000,
                    "dur": dur / 1000,
                }
        raise RuntimeError("decode error")

    schedule = []
    for task in tasks:
        tid = task["task_id"]
        schedule.append({
            "task_id": tid,
            "vae": pick_stage(tid, vae_vars),
            "dit": pick_stage(tid, dit_vars),
        })

    return {
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        "optimal_makespan": solver.Value(makespan) / 1000,
        "schedule": schedule,
        "anytime": rec.history if rec else [],
    }

if __name__ == "__main__":
    tasks = [
        {"task_id": 0, "vae": [(2, 1.2), (4, 0.8)], "dit": [(2, 5.0), (4, 3.2)]},
        {"task_id": 1, "vae": [(2, 1.1), (4, 0.75)], "dit": [(2, 4.8), (4, 3.1)]},
        {"task_id": 2, "vae": [(2, 1.3), (4, 0.85)], "dit": [(2, 5.2), (4, 3.3)]},
        {"task_id": 3, "vae": [(2, 1.0), (4, 0.7)], "dit": [(2, 4.6), (4, 2.9)]},
    ]
    out = solve_optimal(tasks, n_gpus=16, time_limit_sec=100, log=False)
    print(out["status"], out["optimal_makespan"])
    for row in out["schedule"]:
        print(row)
