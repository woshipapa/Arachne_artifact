from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple

from .optimal_fixed_sp import solve_fixed_sp_optimal_makespan


# ============================================================
# Precompute optimistic minimums (LB helper)
# ============================================================

def precompute_min_stage(tasks: List[Dict[str, Any]]):
    """
    (tid, stage) -> (min_sp, min_dur_sec)
    """
    out = {}
    for t in tasks:
        tid = int(t["task_id"])
        for stage in ("vae", "dit"):
            sp, dur = min(t[stage], key=lambda x: x[1])
            out[(tid, stage)] = (int(sp), float(dur))
    return out


# ============================================================
# Valid lower bound: work / gpus  OR  critical chain
# ============================================================

def lb_work_chain(
    tasks: List[Dict[str, Any]],
    gene_to_key: List[Tuple[int, str]],
    partial: List[int],
    min_stage,
    n_gpus: int,
) -> float:
    """
    LB = max(
        sum(sp * dur) / n_gpus,
        max_t (min_vae(t) + min_dit(t))
    )
    """
    id2task = {int(t["task_id"]): t for t in tasks}

    fixed = set()
    total_work = 0.0

    # fixed genes
    for gi, sp in enumerate(partial):
        tid, stage = gene_to_key[gi]
        dur = float(dict(id2task[tid][stage])[int(sp)])
        total_work += float(sp) * dur
        fixed.add((tid, stage))

    # optimistic for unfixed
    for (tid, stage), (msp, mdur) in min_stage.items():
        if (tid, stage) not in fixed:
            total_work += float(msp) * float(mdur)

    work_lb = total_work / float(n_gpus)

    chain_lb = 0.0
    for t in tasks:
        tid = int(t["task_id"])
        chain_lb = max(
            chain_lb,
            min_stage[(tid, "vae")][1] + min_stage[(tid, "dit")][1],
        )

    return max(work_lb, chain_lb)


# ============================================================
# Branch-and-Bound Oracle (Exact, Anytime)
# ============================================================

def bnb_oracle_sp_v1_exact(
    tasks: List[Dict[str, Any]],
    n_gpus: int,
    time_budget_sec: float = 60.0,
    leaf_cp_time_limit_sec: float = 5.0,
    log: bool = False,
) -> Dict[str, Any]:
    """
    Exact oracle:
      - Outer: BnB over SP combinations
      - Inner: CP-SAT solves fixed-SP optimal schedule

    Anytime behavior:
      - Records FIRST feasible solution
      - Records every UB improvement
      - Returns proven_optimal if finished
    """

    t0 = time.time()

    # -----------------------------
    # Setup
    # -----------------------------
    id2task = {int(t["task_id"]): t for t in tasks}

    gene_to_key: List[Tuple[int, str]] = []
    sp_options: List[List[int]] = []

    for t in tasks:
        tid = int(t["task_id"])
        for stage in ("vae", "dit"):
            gene_to_key.append((tid, stage))
            sp_options.append([int(sp) for sp, _ in t[stage]])

    min_stage = precompute_min_stage(tasks)

    best_ub = float("inf")
    best_chrom: List[int] | None = None

    anytime: List[Dict[str, Any]] = []

    nodes = 0
    pruned = 0
    timeout = False

    # -------------------------------------------------
    # Intentionally SLOW ordering (to show GA advantage)
    # -------------------------------------------------
    def ordered_options(gi: int) -> List[int]:
        tid, stage = gene_to_key[gi]
        table = dict(id2task[tid][stage])
        opts = sp_options[gi][:]
        # larger duration first => worse first => slower anytime
        opts.sort(key=lambda sp: float(table[sp]), reverse=True)
        return opts

    # -----------------------------
    # DFS
    # -----------------------------
    def dfs(partial: List[int]):
        nonlocal best_ub, best_chrom, timeout, nodes, pruned

        if time.time() - t0 > time_budget_sec:
            timeout = True
            return

        nodes += 1
        gi = len(partial)

        lb = lb_work_chain(
            tasks, gene_to_key, partial, min_stage, n_gpus
        )

        if lb >= best_ub:
            pruned += 1
            return

        # -----------------------------
        # Leaf: exact solve
        # -----------------------------
        if gi == len(gene_to_key):
            out = solve_fixed_sp_optimal_makespan(
                tasks=tasks,
                n_gpus=n_gpus,
                chromosome=partial,
                time_limit_sec=leaf_cp_time_limit_sec,
                log=False,
                record_anytime=False,
            )

            if out["status"] in ("OPTIMAL", "FEASIBLE") and out["makespan_sec"] is not None:
                mk = float(out["makespan_sec"])

                is_first = best_chrom is None
                is_improve = mk < best_ub

                if is_first or is_improve:
                    best_ub = mk
                    best_chrom = partial[:]
                    anytime.append({
                        "t_sec": time.time() - t0,
                        "makespan_sec": mk,
                        "nodes": nodes,
                        "pruned": pruned,
                        "type": "first" if is_first else "improve",
                    })
            return

        # -----------------------------
        # Branch
        # -----------------------------
        for sp in ordered_options(gi):
            dfs(partial + [sp])
            if timeout:
                return

    dfs([])

    # -----------------------------
    # Normalize gap for plotting
    # -----------------------------
    if best_ub < float("inf"):
        for r in anytime:
            r["gap"] = (r["makespan_sec"] - best_ub) / best_ub

    return {
        "best_makespan": None if best_chrom is None else best_ub,
        "best_chromosome": best_chrom,
        "anytime": anytime,
        "proven_optimal": (not timeout) and (best_chrom is not None),
        "wall_time_sec": time.time() - t0,
        "stats": {
            "nodes": nodes,
            "pruned": pruned,
            "num_genes": len(gene_to_key),
        },
    }
