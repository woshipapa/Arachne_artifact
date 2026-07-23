import os
import json
from typing import Dict, Any, Tuple, List


# ==============================
# Config
# ==============================
DEBUG_PRINT_PER_RANK = True
DEBUG_PRINT_STAGES_ON_ERROR = True
EPS = 1e-9


def _iter_keys_sorted(analysis_data: Dict[str, Any]) -> List[str]:
    """Sort keys like iteration_0, iteration_1, ..."""
    def key_fn(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 10**18
    return sorted([k for k in analysis_data.keys() if k.startswith("iteration_")], key=key_fn)


def _rank_keys_sorted(iter_dict: Dict[str, Any]) -> List[str]:
    def key_fn(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 10**18
    return sorted([k for k in iter_dict.keys() if k.startswith("rank_")], key=key_fn)


def _sum_active_from_stage_analysis(rank_dict: Dict[str, Any]) -> float:
    """
    active = sum(stage_analysis[*].average_duration_s)
    """
    stage_analysis = rank_dict.get("stage_analysis", [])
    total = 0.0
    for s in stage_analysis:
        total += float(s.get("average_duration_s", 0.0))
    return total


def compute_util_and_idle_from_analysis_summary(
    analysis_summary_path: str,
    dump_util_json: bool = True,
    dump_idle_json: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(analysis_summary_path, "r") as f:
        analysis_data = json.load(f)

    util_summary: Dict[str, float] = {}
    idle_summary: Dict[str, float] = {}

    iter_keys = _iter_keys_sorted(analysis_data)
    if not iter_keys:
        raise RuntimeError(f"No iteration_* keys found in: {analysis_summary_path}")

    print("=" * 80)
    print(f"[LOAD] {analysis_summary_path}")
    print(f"[ITERS] {len(iter_keys)} iterations")
    print("=" * 80)

    for iter_key in iter_keys:
        it = analysis_data[iter_key]
        makespan = float(it.get("iteration_total_time_s", 0.0))

        if makespan <= EPS:
            print(f"[WARN] {iter_key}: makespan missing/zero -> {makespan}. Skip.")
            continue

        rank_keys = _rank_keys_sorted(it)
        if not rank_keys:
            print(f"[WARN] {iter_key}: no rank_* keys. Skip.")
            continue

        per_rank_utils = []
        per_rank_idles = []

        print(f"\n--- {iter_key} ---")
        print(f"makespan = {makespan:.6f}s, num_ranks={len(rank_keys)}")

        # Debug gather
        any_error = False
        error_msgs = []

        for rk in rank_keys:
            rdict = it[rk]

            active_from_stages = _sum_active_from_stage_analysis(rdict)

            json_active = rdict.get("active_time_s", None)
            json_idle = rdict.get("idle_time_s", None)

            idle = max(0.0, makespan - active_from_stages)
            util = active_from_stages / makespan if makespan > EPS else 0.0

            per_rank_utils.append(util)
            per_rank_idles.append(idle / makespan)

            if active_from_stages > makespan + 1e-3:
                any_error = True
                error_msgs.append(
                    f"{rk}: active_from_stages({active_from_stages:.6f}) > makespan({makespan:.6f})"
                )

            if util > 1.0 + 1e-3:
                any_error = True
                error_msgs.append(f"{rk}: util({util:.6f}) > 1.0")

            if DEBUG_PRINT_PER_RANK:
                print(
                    f"  {rk}: "
                    f"active(sum stages)={active_from_stages:.6f}s, "
                    f"idle(makespan-active)={idle:.6f}s, "
                    f"util={util*100:.2f}%"
                    + (f", json_active={json_active}" if json_active is not None else "")
                    + (f", json_idle={json_idle}" if json_idle is not None else "")
                )

            if any_error and DEBUG_PRINT_STAGES_ON_ERROR:
                stage_analysis = rdict.get("stage_analysis", [])
                print(f"    [STAGES] {rk} has {len(stage_analysis)} stages:")
                for s in stage_analysis:
                    print(
                        f"      - {s.get('stage_name','?')}: avg={float(s.get('average_duration_s',0.0)):.6f}s"
                    )

        avg_util = sum(per_rank_utils) / len(per_rank_utils) if per_rank_utils else 0.0
        avg_idle_ratio = 1.0 - avg_util

        print(f"[RESULT] {iter_key}: avg_util={avg_util*100:.2f}% | avg_idle_ratio={avg_idle_ratio*100:.2f}%")

        if any_error:
            print("  [ERROR DETECTED] Possible misalignment / inconsistent timing fields:")
            for msg in error_msgs:
                print(f"    - {msg}")

        util_summary[iter_key] = round(avg_util, 6)
        idle_summary[iter_key] = round(avg_idle_ratio, 6)

    # dump
    out_dir = os.path.dirname(analysis_summary_path)
    if dump_util_json:
        util_path = os.path.join(out_dir, "gpu_utilization_summary.json")
        with open(util_path, "w") as f:
            json.dump(util_summary, f, indent=4)
        print(f"\n[SAVED] utilization -> {util_path}")

    if dump_idle_json:
        idle_path = os.path.join(out_dir, "gpu_idle_ratio_summary.json")
        with open(idle_path, "w") as f:
            json.dump(idle_summary, f, indent=4)
        print(f"[SAVED] idle ratio -> {idle_path}")

    return util_summary, idle_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute per-iteration GPU utilization & idle ratio "
                    "(gpu_idle_ratio_summary.json) from analysis_summary.json."
    )
    # Any ONE of these locates analysis_summary.json (produced by visualize_all_timer_rank.py):
    parser.add_argument("--analysis-summary", default=None,
                        help="Direct path to analysis_summary.json.")
    parser.add_argument("--figures-dir", default=None,
                        help="Directory containing analysis_summary.json (e.g. <log-dir>/figures).")
    parser.add_argument("--log-dir", default=None,
                        help="Directory whose figures/ subdir holds analysis_summary.json "
                             "(same --log-dir you passed to visualize_all_timer_rank.py).")
    # Legacy layout (used only when none of the above is given); defaults reproduce the bundle.
    parser.add_argument("--base-log-dir", default="baseline_log")
    parser.add_argument("--framework", default="megatron-lm")
    parser.add_argument("--model", default="hunyuan")
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--partial", default="129",
                        help="max_frames level, e.g. '129' or '' if absent.")
    args = parser.parse_args()

    if args.analysis_summary:
        analysis_path = args.analysis_summary
    elif args.figures_dir:
        analysis_path = os.path.join(args.figures_dir, "analysis_summary.json")
    elif args.log_dir:
        analysis_path = os.path.join(args.log_dir, "figures", "analysis_summary.json")
    else:
        analysis_path = os.path.join(args.base_log_dir, args.framework, args.model,
                                     args.resolution, args.partial, "figures", "analysis_summary.json")

    assert os.path.exists(analysis_path), f"analysis_summary.json not found: {analysis_path}"

    compute_util_and_idle_from_analysis_summary(analysis_path)
