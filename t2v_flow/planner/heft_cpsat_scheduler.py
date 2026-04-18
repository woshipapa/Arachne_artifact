"""
HEFT + CP-SAT 混合调度器
========================
1. HEFT (Heterogeneous Earliest Finish Time) 为 moldable 二阶段任务生成高质量初始解
2. CP-SAT (Google OR-Tools) 利用 HEFT 上界做精确/近最优求解
3. 拓扑感知 GPU 分配（复用已有 TopologyModel + find_best_gpu_allocation_v6）

输出格式与 genetic_algorithm_schedule_v2 完全兼容: (trace, makespan)
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
#     print("WARNING: CP-SAT 求解器需要 ortools 包。请运行: pip install ortools。将回退到纯 HEFT。")

# ---------------------------------------------------------------------------
# 辅助: 从 tasks_input 构建内部数据结构
# ---------------------------------------------------------------------------

def _build_task_meta(tasks_input: List[Dict]):
    """
    Returns
    -------
    task_ids : list[int]
    sp_options : dict[(task_id, stage)] -> list[int]        有序 SP 列表
    durations  : dict[(task_id, stage, sp)] -> float         执行时间
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
    """每个 (task_id, stage) 的平均执行时间 (跨所有 SP)"""
    avg = {}
    for tid in task_ids:
        for stage in ("VAE", "DIT"):
            sps = sp_options[(tid, stage)]
            avg[(tid, stage)] = sum(durations[(tid, stage, sp)] for sp in sps) / len(sps)
    return avg


def _compute_upward_rank(task_ids, avg_cost):
    """
    upward rank: 从该节点到 DAG 出口的最长路径 (含自身)。
    DAG 结构: 对每个 task_id, VAE -> DIT 有前驱边。
    """
    rank = {}
    for tid in task_ids:
        # DIT 是叶子节点
        rank[(tid, "DIT")] = avg_cost[(tid, "DIT")]
        # VAE 的后继是 DIT
        rank[(tid, "VAE")] = avg_cost[(tid, "VAE")] + rank[(tid, "DIT")]
    return rank


def heft_schedule(
    tasks_input: List[Dict],
    total_gpus: int,
    topology_model=None,
    use_topology: bool = True,
) -> Tuple[List[Tuple], float]:
    """
    HEFT 变体: 支持 moldable parallelism (每个 stage 可选不同 SP)。

    算法:
    1. 计算所有 (task, stage) 的 upward rank
    2. 按 rank 降序排列
    3. 对每个 stage, 枚举所有可行 SP, 选择使 EFT (Earliest Finish Time) 最小的 (SP, GPU组)

    Returns: (trace, makespan) — trace 格式与 GA 输出兼容 (8-tuple)
    """
    task_ids, sp_options, durations, dataset_map = _build_task_meta(tasks_input)
    avg_cost = _compute_avg_cost(task_ids, sp_options, durations)
    rank = _compute_upward_rank(task_ids, avg_cost)

    # 构建所有待调度节点并按 rank 降序排序
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
        # 前驱约束
        if stage == "DIT":
            dep_finish = stage_end_time.get((tid, "VAE"), 0.0)
        else:
            dep_finish = 0.0

        # 枚举所有可行 SP, 选择使 finish time 最小的
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
            raise RuntimeError(f"HEFT: 无法为 task {tid} {stage} 找到可行分配")

        # 更新 timeline
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
#  CP-SAT Solver (用 HEFT 上界 warm-start)
# =====================================================================

def _to_int(val, scale=1000):
    """CP-SAT 只接受整数, 将 float 秒转为 int 毫秒"""
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
    """
    CP-SAT 精确调度器。

    建模:
    - 对每个 (task, stage, sp) 创建 optional interval variable
    - 每个 (task, stage) 恰好选中一个 SP
    - cumulative 约束: 任意时刻 GPU 使用量 <= total_gpus
    - 前驱约束: DIT.start >= VAE.end (同一 task)
    - 目标: minimize makespan
    - 用 HEFT makespan 作为上界 hint

    Returns: (trace, makespan) — 8-tuple trace 格式
    """
    # if not HAS_ORTOOLS:
    #     return None, None

    task_ids, sp_options, durations, dataset_map = _build_task_meta(tasks_input)

    SCALE = 1000  # 秒 -> 毫秒
    # 时间上界: 如果有 HEFT 解用它, 否则用所有任务串行的总时间
    if heft_makespan is not None:
        horizon = _to_int(heft_makespan * 1.05, SCALE)  # 留 5% 余量
    else:
        total_serial = sum(
            max(durations[(tid, st, sp)] for sp in sp_options[(tid, st)])
            for tid in task_ids for st in ("VAE", "DIT")
        )
        horizon = _to_int(total_serial, SCALE)

    model = cp_model.CpModel()
    # solver.parameters.num_search_workers = 1
    # --- 决策变量 ---
    # makespan 变量
    makespan_var = model.new_int_var(0, horizon, "makespan")

    # 存储每个 (tid, stage) 的所有 SP 候选
    # interval_vars[(tid, stage, sp)] = (interval, start, end, presence)
    interval_vars = {}
    # 每个 (tid, stage) 选中的 end 变量 (用于前驱约束)
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

            # 这个 (tid, stage) 的汇总 start/end (跨所有 SP 候选)
            agg_start = model.new_int_var(0, horizon, f"start_{tid}_{stage}")
            agg_end = model.new_int_var(0, horizon, f"end_{tid}_{stage}")

            for sp in sps:
                dur_int = _to_int(durations[(tid, stage, sp)], SCALE)
                if dur_int <= 0:
                    dur_int = 1  # CP-SAT 不接受 0 时长

                suffix = f"{tid}_{stage}_sp{sp}"
                presence = model.new_bool_var(f"pres_{suffix}")
                start = model.new_int_var(0, horizon, f"s_{suffix}")
                end = model.new_int_var(0, horizon, f"e_{suffix}")

                interval = model.new_optional_fixed_size_interval_var(
                    start, dur_int, presence, f"itv_{suffix}"
                )

                interval_vars[(tid, stage, sp)] = (interval, start, end, presence)
                presence_literals.append(presence)

                # 关联 end = start + dur (fixed size interval 自动满足)
                model.add(end == start + dur_int).only_enforce_if(presence)

                # 当选中时, agg_start/end 等于对应候选
                model.add(agg_start == start).only_enforce_if(presence)
                model.add(agg_end == end).only_enforce_if(presence)

                # cumulative: 这个 interval 消耗 sp 个 GPU
                all_intervals.append(interval)
                all_demands.append(sp)

            # 恰好选择一个 SP
            model.add_exactly_one(presence_literals)

            stage_start_vars[key] = agg_start
            stage_end_vars[key] = agg_end

    # --- 约束 ---

    # 1. cumulative: 任意时刻 GPU 使用 <= total_gpus
    model.add_cumulative(all_intervals, all_demands, total_gpus)

    # 2. 前驱约束: DIT 在 VAE 之后
    for tid in task_ids:
        model.add(stage_start_vars[(tid, "DIT")] >= stage_end_vars[(tid, "VAE")])

    # 3. makespan >= 所有 end
    for tid in task_ids:
        model.add(makespan_var >= stage_end_vars[(tid, "DIT")])
        model.add(makespan_var >= stage_end_vars[(tid, "VAE")])

    # --- 目标 ---
    model.minimize(makespan_var)

    # --- HEFT warm-start hint ---
    if heft_makespan is not None:
        model.add_hint(makespan_var, _to_int(heft_makespan, SCALE))

    # --- 求解 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    # 多线程加速
    solver.parameters.num_workers = 8
    # 搜索策略优化
    solver.parameters.linearization_level = 2
    solver.parameters.log_search_progress = True

    print(f"[CP-SAT] 开始求解: {len(task_ids)} tasks, {total_gpus} GPUs, "
          f"horizon={_from_int(horizon, SCALE):.1f}s, time_limit={time_limit_sec}s")

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[CP-SAT] 求解失败, status={solver.status_name(status)}, 回退到 HEFT")
        return None, None

    opt_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    raw_makespan = _from_int(solver.value(makespan_var), SCALE)
    print(f"[CP-SAT] 求解完成: {opt_str}, makespan={raw_makespan:.3f}s, "
          f"wall_time={solver.wall_time:.2f}s")

    # --- 提取 SP 选择 (chromosome) ---
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

    # --- 用 CP-SAT 的 SP 选择, 通过拓扑感知 list scheduling 生成最终 trace ---
    # CP-SAT 只决定了 SP 和时序, GPU 具体分配由 list scheduler 完成
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
    """用 chromosome (SP 选择) 通过拓扑感知 list scheduler 生成最终 trace"""
    from .scheduler import generate_trace_from_chromosome_v2
    trace_raw, makespan = generate_trace_from_chromosome_v2(
        tasks_input, chromosome, total_gpus,
        topology_model=topology_model,
        topology_aware=use_topology,
    )
    # generate_trace_from_chromosome_v2 返回 9-tuple (含 penalty), 需要截断为 8-tuple
    trace = [item[:-1] for item in trace_raw]
    return trace, makespan


# =====================================================================
#  主入口: HEFT + CP-SAT 混合调度
# =====================================================================

def heft_cpsat_schedule(
    tasks_input: List[Dict],
    total_gpus: int,
    machine_size: int = None,
    use_topology: bool = True,
    cpsat_time_limit: float = 60.0,
) -> Tuple[List[Tuple], float]:
    """
    混合调度器入口:
    1. HEFT 快速生成初始解 (毫秒级)
    2. CP-SAT 以 HEFT 上界做精确优化 (秒-分钟级)
    3. 取两者中更优的方案
    4. 用拓扑感知 GPU 分配生成最终 trace

    Parameters
    ----------
    tasks_input : list[dict]
        与 GA 相同的任务格式
    total_gpus : int
        GPU 总数
    machine_size : int
        每台机器的 GPU 数 (用于拓扑模型)
    use_topology : bool
        是否启用拓扑感知
    cpsat_time_limit : float
        CP-SAT 求解时间限制 (秒)

    Returns
    -------
    (trace, makespan) : 与 GA 输出格式完全兼容
    """
    if machine_size is None:
        machine_size = total_gpus

    # 初始化拓扑模型
    try:
        topology_model = TopologyModel(
            params_file="fit_results.json",
            machine_size=machine_size,
            enable_topology=use_topology,
        )
    except Exception as e:
        print(f"TopologyModel 初始化失败: {e}, 使用 dummy 模型")
        topology_model = type("DummyModel", (), {
            "predict_penalty": lambda self, gl: 1.0,
            "machine_size": machine_size,
            "score_bitmask": lambda self, gt: (1.0, 0),
        })()

    # ---- Phase 1: HEFT ----
    print("\n--- Phase 1: HEFT 快速求解 ---")
    t0 = time.time()
    heft_trace, heft_makespan = heft_schedule(
        tasks_input, total_gpus,
        topology_model=topology_model,
        use_topology=use_topology,
    )
    heft_time = time.time() - t0
    print(f"[HEFT] makespan={heft_makespan:.3f}s, 耗时={heft_time:.3f}s")

    # ---- Phase 2: CP-SAT (用 HEFT 上界) ----
    print("\n--- Phase 2: CP-SAT 精确优化 ---")
    t1 = time.time()
    cpsat_chromosome, cpsat_raw_makespan = cpsat_schedule(
        tasks_input, total_gpus,
        heft_makespan=heft_makespan,
        time_limit_sec=cpsat_time_limit,
        topology_model=topology_model,
        use_topology=use_topology,
    )
    cpsat_time = time.time() - t1

    # ---- Phase 3: 选择更优方案并生成最终 trace ----
    if cpsat_chromosome is not None:
        # 用 CP-SAT 的 SP 选择通过拓扑感知 list scheduler 生成 trace
        cpsat_trace, cpsat_makespan = _chromosome_to_trace(
            tasks_input, cpsat_chromosome, total_gpus,
            topology_model, use_topology,
        )
        print(f"[CP-SAT] 拓扑感知 trace makespan={cpsat_makespan:.3f}s, 耗时={cpsat_time:.3f}s")

        if cpsat_makespan <= heft_makespan:
            print(f"\n[HEFT+CP-SAT] 选择 CP-SAT 方案 "
                  f"(改进 {(1 - cpsat_makespan/heft_makespan)*100:.1f}%)")
            return cpsat_trace, cpsat_makespan
        else:
            print(f"\n[HEFT+CP-SAT] CP-SAT 未改进, 选择 HEFT 方案")
            return heft_trace, heft_makespan
    else:
        print(f"\n[HEFT+CP-SAT] CP-SAT 不可用, 使用 HEFT 方案")
        return heft_trace, heft_makespan
