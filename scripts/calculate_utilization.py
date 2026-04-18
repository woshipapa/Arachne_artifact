import os
import json
from typing import Dict, Any, Tuple, List


# ==============================
# Config
# ==============================
DEBUG_PRINT_PER_RANK = True     # True: 打印每个 rank 的 active/idle/util
DEBUG_PRINT_STAGES_ON_ERROR = True  # util>1 或 active>makespan 时，打印该 rank 的 stage 列表
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
    这和你 Gantt 里 current_time += duration_s 的语义一致。
    """
    stage_analysis = rank_dict.get("stage_analysis", [])
    total = 0.0
    for s in stage_analysis:
        # 你写进去的是 round(avg_duration,4)，字段名是 average_duration_s
        total += float(s.get("average_duration_s", 0.0))
    return total


def compute_util_and_idle_from_analysis_summary(
    analysis_summary_path: str,
    dump_util_json: bool = True,
    dump_idle_json: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从 analysis_summary.json 计算：
      - gpu_utilization_summary.json: 每个 iteration 的平均 GPU utilization (0~1)
      - gpu_idle_ratio_summary.json: 每个 iteration 的平均 idle 比例 (0~1), = 1 - util
    """
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

            # 主口径：从 stage_analysis 求和得到 active（与绘图一致）
            active_from_stages = _sum_active_from_stage_analysis(rdict)

            # JSON 里也写了 active_time_s/idle_time_s（可对照）
            json_active = rdict.get("active_time_s", None)
            json_idle = rdict.get("idle_time_s", None)

            idle = max(0.0, makespan - active_from_stages)
            util = active_from_stages / makespan if makespan > EPS else 0.0

            per_rank_utils.append(util)
            per_rank_idles.append(idle / makespan)

            # 检查不合理情况
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

            # 如果出错，打印 stage 列表帮助定位
            if any_error and DEBUG_PRINT_STAGES_ON_ERROR:
                stage_analysis = rdict.get("stage_analysis", [])
                print(f"    [STAGES] {rk} has {len(stage_analysis)} stages:")
                for s in stage_analysis:
                    print(
                        f"      - {s.get('stage_name','?')}: avg={float(s.get('average_duration_s',0.0)):.6f}s"
                    )

        avg_util = sum(per_rank_utils) / len(per_rank_utils) if per_rank_utils else 0.0
        avg_idle_ratio = 1.0 - avg_util  # 等价于 mean(idle/makespan)

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
    # 你保持原来的路径拼接风格即可
    model, resolution = "hunyuan", "720p"
    base_log_dir = "baseline_log"
    # base_log_dir = "dynamic_flex_exp_log"
    # framework = "megatron-lm"  # 或 "megatron-lm" / "deepspeed"
    framework = "megatron-lm"
    partial = "129"           # 如果你有 max_frames 就放这里，比如 "81"；没有就 ""

    output_figure_dir = os.path.join(base_log_dir, framework, model, resolution, partial, "figures")
    analysis_path = os.path.join(output_figure_dir, "analysis_summary.json")

    assert os.path.exists(analysis_path), f"analysis_summary.json not found: {analysis_path}"

    compute_util_and_idle_from_analysis_summary(analysis_path)
