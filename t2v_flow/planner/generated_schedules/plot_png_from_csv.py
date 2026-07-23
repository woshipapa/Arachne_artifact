"""
Replay a scheduling trace CSV and:
1) Generate GPU timeline plot
2) Analyze straggler ranks & final tasks
3) Build iteration execution overview JSON (optional rank tasks)
4) Collect iterations whose final completion is SP=4 (Megatron / FlexSP)
"""

import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import json
import re
from typing import Dict
from collections import defaultdict


# ============================================================
# configuration (new switches go here)
# ============================================================



DUMP_RANK_TASKS = False            # whether to dump each rank's task list
COLLECT_FINAL_SP4 = True          # whether to collect statistics for SP=4 iterations

SP_SIZE_TO_TRACK = 4              # set to 4 here; extensible later

GLOBAL_ITERATION_DATA = {}        # iteration_id -> detail
FINAL_SP4_ITERATIONS = []         # iteration_id list
FINAL_NON_SP4_ITERATIONS = []   # === NEW ===
FLEX_MEGA_SAME_FINAL_TASK_ITERS = []   # === NEW ===
FLEX_MEGA_DIFF_FINAL_TASK_ITERS = []   # === NEW ===

# ============================================================
# core analysis functions (unchanged)
# ============================================================
def extract_final_task_identity(final_tasks):
    """
    Turn final_tasks into a comparable set.
    identity = (Task_ID, Stage, DatasetID)
    """
    return set(
        (t["Task_ID"], t["Stage"], t["DatasetID"])
        for t in final_tasks
    )

def load_and_analyze_trace(csv_path: str) -> Dict:
    """
    Read one trace CSV and return the result of analyze_straggler_from_csv directly.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    if isinstance(df["GPU_List"].iloc[0], str):
        df["GPU_List"] = df["GPU_List"].apply(ast.literal_eval)

    return analyze_straggler_from_csv(df)


def build_cross_schedule_compare(
    iteration_id: str,
    model_type: str,
    resolution: str,
    max_frames: int,
    primary_schedule: str,
    baseline_schedule: str
) -> Dict:
    """
    Compare the same iteration side by side under two schedules.
    """
    primary_csv = (
        f"{model_type}/{resolution}/{max_frames}/"
        f"{primary_schedule}/schedule_{iteration_id}_trace.csv"
    )
    baseline_csv = (
        f"{model_type}/{resolution}/{max_frames}/"
        f"{baseline_schedule}/schedule_{iteration_id}_trace.csv"
    )

    primary = load_and_analyze_trace(primary_csv)
    baseline = load_and_analyze_trace(baseline_csv)

    if primary is None or baseline is None:
        return None

    return {
        primary_schedule: {
            "makespan": primary["makespan"],
            "final_ranks": primary["final_ranks"],
            "final_tasks": primary["final_tasks"],
        },
        baseline_schedule: {
            "makespan": baseline["makespan"],
            "final_ranks": baseline["final_ranks"],
            "final_tasks": baseline["final_tasks"],
        },
        "makespan_speedup": (
            baseline["makespan"] / primary["makespan"]
            if primary["makespan"] > 0 else None
        )
    }


def analyze_straggler_from_csv(df: pd.DataFrame) -> Dict:
    makespan = df["End"].max()
    final_tasks = df[df["End"] == makespan]

    straggler_info = []

    for _, row in final_tasks.iterrows():
        gpus = row["GPU_List"]
        if isinstance(gpus, str):
            gpus = ast.literal_eval(gpus)

        straggler_info.append({
            "Task_ID": int(row["Task_ID"]),
            "Stage": row["Stage"],
            "DatasetID": row["DatasetID"],
            "GPUs": gpus,
            "Start": float(row["Start"]),
            "End": float(row["End"]),
            "Duration": float(row["Time"]),
        })

    straggler_ranks = sorted(
        set(gpu for t in straggler_info for gpu in t["GPUs"])
    )

    return {
        "makespan": float(makespan),
        "final_ranks": straggler_ranks,
        "final_tasks": straggler_info,
    }


# ============================================================
# per-rank execution view (optional output)
# ============================================================

def build_rank_execution_view(df: pd.DataFrame) -> Dict:
    if isinstance(df["GPU_List"].iloc[0], str):
        df["GPU_List"] = df["GPU_List"].apply(ast.literal_eval)

    exploded = df.explode("GPU_List", ignore_index=True)
    exploded = exploded.rename(columns={"GPU_List": "Rank"})

    rank_tasks = defaultdict(list)

    for row in exploded.itertuples(index=False):
        rank_tasks[str(int(row.Rank))].append({
            "Task_ID": int(row.Task_ID),
            "Stage": row.Stage,
            "DatasetID": row.DatasetID,
            "Start": float(row.Start),
            "End": float(row.End),
            "Duration": float(row.Time),
        })

    for r in rank_tasks:
        rank_tasks[r].sort(key=lambda x: x["Start"])

    return {
        "num_ranks": len(rank_tasks),
        "ranks": dict(rank_tasks)
    }


# ============================================================
# main plotting and summary (the enhancements live here)
# ============================================================

def plot_schedule_from_csv(
    csv_path: str,
    total_gpus: int,
    save_path: str = None,
    dump_audit_json: bool = True,
    schedule_type: str = ""
):
    global GLOBAL_ITERATION_DATA, FINAL_SP4_ITERATIONS

    df = pd.read_csv(csv_path)

    # the iteration id comes from the filename
    m = re.search(r"schedule_(\d+)_trace\.csv", csv_path)
    iteration_id = m.group(1) if m else os.path.basename(csv_path)

    # -------------------------------
    # straggler analysis
    # -------------------------------
    audit = analyze_straggler_from_csv(df)

    # -------------------------------
    # SP=4 iteration statistics
    # -------------------------------
    if COLLECT_FINAL_SP4 and schedule_type in {"megatron-lm", "flex_sp"}:
        if len(audit["final_ranks"]) == SP_SIZE_TO_TRACK:
            FINAL_SP4_ITERATIONS.append(iteration_id)
        else:
            FINAL_NON_SP4_ITERATIONS.append(iteration_id)


    # -------------------------------
    # per-rank execution view (optional)
    # -------------------------------
    iteration_entry = {
        "makespan": audit["makespan"],
        "final_ranks": audit["final_ranks"],
        "final_tasks": audit["final_tasks"],
    }
    if schedule_type == "flex_sp":
        cross_cmp = build_cross_schedule_compare(
            iteration_id=iteration_id,
            model_type=MODEL_TYPE,
            resolution=RESOLUTION,
            max_frames=MAX_FRAMES,
            primary_schedule="flex_sp",
            baseline_schedule="megatron-lm"
        )
    # ============================================================
# FlexSP vs Megatron final-task consistency check
# ============================================================

    if schedule_type == "flex_sp" and cross_cmp is not None:
        flex_tasks = extract_final_task_identity(
            cross_cmp["flex_sp"]["final_tasks"]
        )
        mega_tasks = extract_final_task_identity(
            cross_cmp["megatron-lm"]["final_tasks"]
        )

        if flex_tasks == mega_tasks:
            FLEX_MEGA_SAME_FINAL_TASK_ITERS.append(iteration_id)
        else:
            FLEX_MEGA_DIFF_FINAL_TASK_ITERS.append(iteration_id)


    if cross_cmp is not None:
        iteration_entry["cross_schedule_compare"] = cross_cmp

    if DUMP_RANK_TASKS:
        rank_view = build_rank_execution_view(df)
        iteration_entry["num_ranks"] = rank_view["num_ranks"]
        iteration_entry["ranks"] = rank_view["ranks"]

    GLOBAL_ITERATION_DATA[iteration_id] = iteration_entry

    # -------------------------------
    # audit JSON (kept as before)
    # -------------------------------
    if dump_audit_json:
        audit_path = csv_path.replace(".csv", "_straggler_audit.json")
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

    # -------------------------------
    # original GPU timeline plot (kept in full)
    # -------------------------------
    if isinstance(df["GPU_List"].iloc[0], str):
        df["GPU_List"] = df["GPU_List"].apply(ast.literal_eval)

    plot_df = df.explode("GPU_List", ignore_index=True)
    plot_df = plot_df.rename(columns={"GPU_List": "GPU"})

    plot_df["Label"] = (
        plot_df["Stage"].astype(str)
        + "-T"
        + plot_df["Task_ID"].astype(str)
        + "["
        + plot_df["DatasetID"].astype(str)
        + "]"
    )

    plt.figure(figsize=(16, max(8, total_gpus * 0.6)))
    colors = {"DIT": "deepskyblue", "VAE": "salmon"}

    for row in plot_df.itertuples(index=False):
        plt.barh(
            y=row.GPU,
            width=row.Time,
            left=row.Start,
            color=colors.get(row.Stage, "grey"),
            edgecolor="black",
            height=0.7,
        )

    plt.yticks(range(total_gpus), [f"GPU {i}" for i in range(total_gpus)])
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Rank")
    plt.title("GPU Usage Timeline (Replayed from Trace)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


# ============================================================
    # main (same path construction as before)
# ============================================================

if __name__ == "__main__":
    iter_list = range(0, 60)

    SCHEDULE_TYPE = "flex_sp"   # megatron-lm / flex_sp
    MODEL_TYPE = "hunyuan"
    RESOLUTION = "720p"
    MAX_FRAMES = 129
    TOTAL_GPUS = 16
    ITERATION_OVERVIEW_JSON = f"iteration_execution_overview_{MODEL_TYPE}_{RESOLUTION}_{MAX_FRAMES}_{SCHEDULE_TYPE}.json"
    for i in iter_list:
        TRACE_FILE = f"{MODEL_TYPE}/{RESOLUTION}/{MAX_FRAMES}/{SCHEDULE_TYPE}/schedule_{i}_trace.csv"
        OUTPUT_IMAGE_PATH = TRACE_FILE.replace(".csv", ".png")

        if not os.path.exists(TRACE_FILE):
            continue

        plot_schedule_from_csv(
            csv_path=TRACE_FILE,
            total_gpus=TOTAL_GPUS,
            save_path=OUTPUT_IMAGE_PATH,
            dump_audit_json=True,
            schedule_type=SCHEDULE_TYPE
        )

    # ========================================================
    # finally export the overview JSON
    # ========================================================

    final_json = {
     "meta": {
    "schedule_type": SCHEDULE_TYPE,
    "dump_rank_tasks": DUMP_RANK_TASKS,

    "sp4_final_iterations": sorted(FINAL_SP4_ITERATIONS),
    "non_sp4_final_iterations": sorted(FINAL_NON_SP4_ITERATIONS),

    # === NEW ===
    "flex_vs_megatron_same_final_task_iterations":
        sorted(FLEX_MEGA_SAME_FINAL_TASK_ITERS),
    "flex_vs_megatron_diff_final_task_iterations":
        sorted(FLEX_MEGA_DIFF_FINAL_TASK_ITERS),
},


        "iterations": GLOBAL_ITERATION_DATA
    }

    with open(ITERATION_OVERVIEW_JSON, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"\n[SUCCESS] Iteration execution overview saved to {ITERATION_OVERVIEW_JSON}")
