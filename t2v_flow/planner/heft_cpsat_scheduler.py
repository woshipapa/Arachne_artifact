"""
========================

"""

import time
from typing import List, Dict, Tuple, Optional
from .TopologyModel import TopologyModel
import faulthandler
faulthandler.enable()

# HAS_ORTOOLS = False
# try:
#     from ortools.sat.python import cp_model
#     HAS_ORTOOLS = True
# except ImportError:

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _build_task_meta(tasks_input: List[Dict]):
    """
    Returns
    -------
    task_ids : list[int]
    dataset_map: dict[task_id] -> str
    """
    task_ids = []
    sp_options = {}
    durations = {}
    dataset_map = {}
    for t in tasks_input:
        tid = t["task_id"]
        task_ids.append(tid)
        dataset_map[tid] = t["dataset_id"]
        for stage, key in [("VAE", "vae"), ("DIT", "dit")]:
            sps = []
            for sp, dur in t[key]:
                sps.append(sp)
                durations[(tid, stage, sp)] = dur
            sp_options[(tid, stage)] = sps
    return task_ids, sp_options, durations, dataset_map


# =====================================================================
#  HEFT for Moldable Two-Stage Tasks
# =====================================================================

def _compute_avg_cost(task_ids, sp_options, durations):
    avg = {}
    for tid in task_ids:
        for stage in ("VAE", "DIT"):
            sps = sp_options[(tid, stage)]
            avg[(tid, stage)] = sum(durations[(tid, stage, sp)] for sp in sps) / len(sps)
    return avg


def _compute_upward_rank(task_ids, avg_cost):
    rank = {}
    for tid in task_ids:
        rank[(tid, "DIT")] = avg_cost[(tid, "DIT")]
        rank[(tid, "VAE")] = avg_cost[(tid, "VAE")] + rank[(tid, "DIT")]
    return rank


def heft_schedule(
    tasks_input: List[Dict],
    total_gpus: int,
    topology_model=None,
    use_topology: bool = True,
) -> Tuple[List[Tuple], float]:
    task_ids, sp_options, durations, dataset_map = _build_task_meta(tasks_input)
    avg_cost = _compute_avg_cost(task_ids, sp_options, durations)
    rank = _compute_upward_rank(task_ids, avg_cost)

    all_nodes = []
    for tid in task_ids:
        all_nodes.append((tid, "VAE"))
        all_nodes.append((tid, "DIT"))
    all_nodes.sort(key=lambda x: rank[x], reverse=True)

    gpu_timeline = [0.0] * total_gpus
    stage_end_time = {}   # (tid, stage) -> end_time
    trace = []

    if use_topology and topology_model is not None:
        from .scheduler import find_best_gpu_allocation_v6
        alloc_fn = lambda gt, k, dur, est: find_best_gpu_allocation_v6(
            gt, k, dur, earliest_start_time=est,
            topology_model=topology_model,
            lookahead_threshold=5, beam_width=5,
        )
    else:
        from .scheduler import find_best_gpu_allocation_resource_only
        alloc_fn = lambda gt, k, dur, est: find_best_gpu_allocation_resource_only(
            gt, k, dur, earliest_start_time=est,
        )

    for (tid, stage) in all_nodes:
        if stage == "DIT":
            dep_finish = stage_end_time.get((tid, "VAE"), 0.0)
        else:
            dep_finish = 0.0

        best_sp = None
        best_gpus = None
        best_start = None
        best_end = float("inf")
        best_penalty = 1.0

        for sp in sp_options[(tid, stage)]:
            if sp > total_gpus:
                continue
            dur = durations[(tid, stage, sp)]
            gpus, start, end, penalty = alloc_fn(gpu_timeline, sp, dur, dep_finish)
            if gpus is not None and end < best_end:
                best_sp = sp
                best_gpus = gpus
                best_start = start
                best_end = end
                best_penalty = penalty

        if best_sp is None:
            raise RuntimeError(f"HEFT: no feasible assignment for task {tid} {stage}")

        for g in best_gpus:
            gpu_timeline[g] = best_end

        stage_end_time[(tid, stage)] = best_end
        trace.append((
            tid, stage, best_sp,
            best_end - best_start,
            best_start, best_end,
            dataset_map[tid],
            best_gpus,
        ))

    makespan = max(gpu_timeline)
    return trace, makespan


# =====================================================================
# =====================================================================

def _to_int(val, scale=1000):
    return int(round(val * scale))

def _from_int(val, scale=1000):
    return val / scale


def cpsat_schedule(
    tasks_input: List[Dict],
    total_gpus: int,
    heft_makespan: float = None,
    time_limit_sec: float = 60.0,
    topology_model=None,
    use_topology: bool = True,
) -> Tuple[List[Tuple], float]:
    # if not HAS_ORTOOLS:
    #     return None, None

    task_ids, sp_options, durations, dataset_map = _build_task_meta(tasks_input)

    SCALE = 1000
    if heft_makespan is not None:
        horizon = _to_int(heft_makespan * 1.05, SCALE)
    else:
        total_serial = sum(
            max(durations[(tid, st, sp)] for sp in sp_options[(tid, st)])
            for tid in task_ids for st in ("VAE", "DIT")
        )
        horizon = _to_int(total_serial, SCALE)

    model = cp_model.CpModel()
    # solver.parameters.num_search_workers = 1
    makespan_var = model.new_int_var(0, horizon, "makespan")

    # interval_vars[(tid, stage, sp)] = (interval, start, end, presence)
    interval_vars = {}
    stage_end_vars = {}
    stage_start_vars = {}
    # cumulative demands
    all_intervals = []
    all_demands = []

    for tid in task_ids:
        for stage in ("VAE", "DIT"):
            key = (tid, stage)
            sps = sp_options[key]
            presence_literals = []

            agg_start = model.new_int_var(0, horizon, f"start_{tid}_{stage}")
            agg_end = model.new_int_var(0, horizon, f"end_{tid}_{stage}")

            for sp in sps:
                dur_int = _to_int(durations[(tid, stage, sp)], SCALE)
                if dur_int <= 0:
                    dur_int = 1

                suffix = f"{tid}_{stage}_sp{sp}"
                presence = model.new_bool_var(f"pres_{suffix}")
                start = model.new_int_var(0, horizon, f"s_{suffix}")
                end = model.new_int_var(0, horizon, f"e_{suffix}")

                interval = model.new_optional_fixed_size_interval_var(
                    start, dur_int, presence, f"itv_{suffix}"
                )

                interval_vars[(tid, stage, sp)] = (interval, start, end, presence)
                presence_literals.append(presence)

                model.add(end == start + dur_int).only_enforce_if(presence)

                model.add(agg_start == start).only_enforce_if(presence)
                model.add(agg_end == end).only_enforce_if(presence)

                all_intervals.append(interval)
                all_demands.append(sp)

            model.add_exactly_one(presence_literals)

            stage_start_vars[key] = agg_start
            stage_end_vars[key] = agg_end


    model.add_cumulative(all_intervals, all_demands, total_gpus)

    for tid in task_ids:
        model.add(stage_start_vars[(tid, "DIT")] >= stage_end_vars[(tid, "VAE")])

    for tid in task_ids:
        model.add(makespan_var >= stage_end_vars[(tid, "DIT")])
        model.add(makespan_var >= stage_end_vars[(tid, "VAE")])

    model.minimize(makespan_var)

    # --- HEFT warm-start hint ---
    if heft_makespan is not None:
        model.add_hint(makespan_var, _to_int(heft_makespan, SCALE))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_workers = 8
    solver.parameters.linearization_level = 2
    solver.parameters.log_search_progress = True

    print(f"[CP-SAT] solving: {len(task_ids)} tasks, {total_gpus} GPUs, "
          f"horizon={_from_int(horizon, SCALE):.1f}s, time_limit={time_limit_sec}s")

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[CP-SAT] solve failed, status={solver.status_name(status)}, falling back to HEFT")
        return None, None

    opt_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    raw_makespan = _from_int(solver.value(makespan_var), SCALE)
    print(f"[CP-SAT] solve complete: {opt_str}, makespan={raw_makespan:.3f}s, "
          f"wall_time={solver.wall_time:.2f}s")

    chosen_sp = {}  # (tid, stage) -> sp
    chosen_start = {}
    chosen_end = {}
    for tid in task_ids:
        for stage in ("VAE", "DIT"):
            for sp in sp_options[(tid, stage)]:
                _, s_var, e_var, p_var = interval_vars[(tid, stage, sp)]
                if solver.value(p_var):
                    chosen_sp[(tid, stage)] = sp
                    chosen_start[(tid, stage)] = _from_int(solver.value(s_var), SCALE)
                    chosen_end[(tid, stage)] = _from_int(solver.value(e_var), SCALE)
                    break

    chromosome = []
    for t in tasks_input:
        tid = t["task_id"]
        chromosome.append(chosen_sp[(tid, "VAE")])
        chromosome.append(chosen_sp[(tid, "DIT")])

    return chromosome, raw_makespan


def _chromosome_to_trace(
    tasks_input: List[Dict],
    chromosome: List[int],
    total_gpus: int,
    topology_model,
    use_topology: bool,
) -> Tuple[List[Tuple], float]:
    from .scheduler import generate_trace_from_chromosome_v2
    trace_raw, makespan = generate_trace_from_chromosome_v2(
        tasks_input, chromosome, total_gpus,
        topology_model=topology_model,
        topology_aware=use_topology,
    )
    trace = [item[:-1] for item in trace_raw]
    return trace, makespan


# =====================================================================
# =====================================================================

def heft_cpsat_schedule(
    tasks_input: List[Dict],
    total_gpus: int,
    machine_size: int = None,
    use_topology: bool = True,
    cpsat_time_limit: float = 60.0,
) -> Tuple[List[Tuple], float]:
    """

    Parameters
    ----------
    tasks_input : list[dict]
    total_gpus : int
    machine_size : int
    use_topology : bool
    cpsat_time_limit : float

    Returns
    -------
    """
    if machine_size is None:
        machine_size = total_gpus

    try:
        topology_model = TopologyModel(
            params_file="fit_results.json",
            machine_size=machine_size,
            enable_topology=use_topology,
        )
    except Exception as e:
        print(f"TopologyModel failed to initialise: {e}; using a dummy model")
        topology_model = type("DummyModel", (), {
            "predict_penalty": lambda self, gl: 1.0,
            "machine_size": machine_size,
            "score_bitmask": lambda self, gt: (1.0, 0),
        })()

    # ---- Phase 1: HEFT ----
    print("\n--- Phase 1: fast HEFT solve ---")
    t0 = time.time()
    heft_trace, heft_makespan = heft_schedule(
        tasks_input, total_gpus,
        topology_model=topology_model,
        use_topology=use_topology,
    )
    heft_time = time.time() - t0
    print(f"[HEFT] makespan={heft_makespan:.3f}s, elapsed={heft_time:.3f}s")

    print("\n--- Phase 2: exact CP-SAT optimisation ---")
    t1 = time.time()
    cpsat_chromosome, cpsat_raw_makespan = cpsat_schedule(
        tasks_input, total_gpus,
        heft_makespan=heft_makespan,
        time_limit_sec=cpsat_time_limit,
        topology_model=topology_model,
        use_topology=use_topology,
    )
    cpsat_time = time.time() - t1

    if cpsat_chromosome is not None:
        cpsat_trace, cpsat_makespan = _chromosome_to_trace(
            tasks_input, cpsat_chromosome, total_gpus,
            topology_model, use_topology,
        )
        print(f"[CP-SAT] topology-aware trace makespan={cpsat_makespan:.3f}s, elapsed={cpsat_time:.3f}s")

        if cpsat_makespan <= heft_makespan:
            print(f"\n[HEFT+CP-SAT] choosing the CP-SAT plan "
                  f"(improvement {(1 - cpsat_makespan/heft_makespan)*100:.1f}%)")
            return cpsat_trace, cpsat_makespan
        else:
            print(f"\n[HEFT+CP-SAT] CP-SAT did not improve; choosing the HEFT plan")
            return heft_trace, heft_makespan
    else:
        print(f"\n[HEFT+CP-SAT] CP-SAT unavailable; using the HEFT plan")
        return heft_trace, heft_makespan
