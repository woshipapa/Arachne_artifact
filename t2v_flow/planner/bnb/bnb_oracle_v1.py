import time
from typing import List, Dict, Any, Tuple, Optional


def _build_tables(tasks: List[Dict[str, Any]]):
    """
    Build fast lookup tables:
      - task_by_id[tid] -> task dict
      - dur[(tid, stage, sp)] -> duration (seconds, float)
      - sp_options_per_gene: list[list[int]]
      - gene_to_taskstage: list[(tid, stage)]
      - min_stage[(tid, stage)] -> (min_sp, min_dur)
    """
    task_by_id = {int(t["task_id"]): t for t in tasks}

    dur: Dict[Tuple[int, str, int], float] = {}
    min_stage: Dict[Tuple[int, str], Tuple[int, float]] = {}

    gene_to_taskstage: List[Tuple[int, str]] = []
    sp_options_per_gene: List[List[int]] = []

    # Keep gene order consistent with your GA v1:
    # [task0 VAE, task0 DIT, task1 VAE, task1 DIT, ...]
    for t in tasks:
        tid = int(t["task_id"])
        for stage in ("vae", "dit"):
            opts = [(int(sp), float(sec)) for sp, sec in t[stage]]
            if not opts:
                raise ValueError(f"Task {tid} stage {stage} has no options.")

            for sp, sec in opts:
                dur[(tid, stage, sp)] = sec

            best_sp, best_sec = min(opts, key=lambda x: x[1])
            min_stage[(tid, stage)] = (best_sp, best_sec)

            gene_to_taskstage.append((tid, stage))
            sp_options_per_gene.append([sp for sp, _ in opts])

    return task_by_id, dur, min_stage, gene_to_taskstage, sp_options_per_gene


def compute_lb(
    partial_chrom: List[int],
    gene_to_taskstage: List[Tuple[int, str]],
    dur: Dict[Tuple[int, str, int], float],
    min_stage: Dict[Tuple[int, str], Tuple[int, float]],
    n_gpus: int,
    tasks: List[Dict[str, Any]],
) -> float:
    """
    Weak-but-valid lower bound:
      LB = max( work_lb, chain_lb )

    work_lb = (fixed_work + optimistic_remaining_work) / n_gpus
      where work = sum(sp * dur)

    chain_lb = max over tasks of (min_vae_dur + min_dit_dur)
    """
    total_work = 0.0
    fixed = set()

    # fixed part
    for gi, sp in enumerate(partial_chrom):
        tid, stage = gene_to_taskstage[gi]
        d = dur[(tid, stage, int(sp))]
        total_work += float(sp) * d
        fixed.add((tid, stage))

    # optimistic remaining part
    for (tid, stage), (min_sp, min_dur) in min_stage.items():
        if (tid, stage) not in fixed:
            total_work += float(min_sp) * float(min_dur)

    work_lb = total_work / float(n_gpus)

    # per-task chain lb (precedence within each task)
    chain_lb = 0.0
    for t in tasks:
        tid = int(t["task_id"])
        chain_lb = max(chain_lb, float(min_stage[(tid, "vae")][1] + min_stage[(tid, "dit")][1]))

    return max(work_lb, chain_lb)


def bnb_oracle_v1(
    tasks: List[Dict[str, Any]],
    n_gpus: int,
    time_budget_sec: float,
    generate_trace_fn,  # generate_trace_from_chromosome_v1(tasks, chrom, total_gpus) -> (trace, makespan)
    *,
    record_anytime: bool = True,
    seed_ub_from: Optional[List[int]] = None,   # optionally provide an initial chromosome to set a UB
    branch_order: str = "as_is",                # "as_is" | "random" (kept simple)
):
    """
    Exact branch-and-bound over SP chromosomes (topology-agnostic),
    with controllable anytime behavior.

    Returns dict:
      {
        "best_makespan": float,
        "best_chromosome": List[int] | None,
        "anytime": [{"t":..., "makespan":...}, ...],
        "proven_optimal": bool,
        "wall_time": float,
        "stats": {...}
      }
    """
    if n_gpus <= 0:
        raise ValueError("n_gpus must be positive.")
    if time_budget_sec <= 0:
        raise ValueError("time_budget_sec must be positive.")

    task_by_id, dur, min_stage, gene_to_taskstage, sp_options_per_gene = _build_tables(tasks)
    num_genes = len(gene_to_taskstage)

    best_ub = float("inf")
    best_chrom = None
    anytime = []

    # stats
    nodes = 0
    prunes = 0
    leaves = 0

    t0 = time.time()
    timeout = False

    # Optional: seed a starting UB using a provided chromosome
    if seed_ub_from is not None:
        if len(seed_ub_from) != num_genes:
            raise ValueError(f"seed_ub_from has len={len(seed_ub_from)} but expected {num_genes}")
        _, ms = generate_trace_fn(tasks, seed_ub_from, n_gpus)
        best_ub = float(ms)
        best_chrom = seed_ub_from[:]
        if record_anytime:
            anytime.append({"t": 0.0, "makespan": best_ub})

    # DFS
    def dfs(partial_chrom: List[int]):
        nonlocal best_ub, best_chrom, timeout, nodes, prunes, leaves

        # budget check
        if (time.time() - t0) > time_budget_sec:
            timeout = True
            return

        nodes += 1
        gi = len(partial_chrom)

        lb = compute_lb(
            partial_chrom=partial_chrom,
            gene_to_taskstage=gene_to_taskstage,
            dur=dur,
            min_stage=min_stage,
            n_gpus=n_gpus,
            tasks=tasks,
        )

        if lb >= best_ub:
            prunes += 1
            return

        # leaf => exact evaluation with your v1 scheduler
        if gi == num_genes:
            leaves += 1
            _, makespan = generate_trace_fn(tasks, partial_chrom, n_gpus)
            makespan = float(makespan)
            if makespan < best_ub:
                best_ub = makespan
                best_chrom = partial_chrom[:]
                if record_anytime:
                    anytime.append({"t": time.time() - t0, "makespan": best_ub})
            return

        # branch
        opts = sp_options_per_gene[gi]
        if branch_order == "random":
            # deterministic-ish shuffle based on time (simple; you can inject RNG if you want)
            opts = opts[:]  # copy
            opts.sort(key=lambda x: hash((gi, x)) % 10)

        for sp in opts:
            dfs(partial_chrom + [sp])
            if timeout:
                return

    dfs([])

    return {
        "best_makespan": best_ub,
        "best_chromosome": best_chrom,
        "anytime": anytime,
        "proven_optimal": not timeout,
        "wall_time": time.time() - t0,
        "stats": {
            "nodes": nodes,
            "prunes": prunes,
            "leaves": leaves,
            "num_genes": num_genes,
        },
    }
