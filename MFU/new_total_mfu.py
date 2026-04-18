import yaml
import math
import numpy as np
from collections import defaultdict, deque
import argparse
import os
import sys
import json, glob, re
import pandas as pd

from .calculate_dit_mfu import estimate_hunyuan_dit_train_flops
from .calculate_vae_mfu import estimate_hunyuan_vae_encoder_flops


# ==============================================================================
# 模块 1: Task Name 解析
# ==============================================================================

TASK_NAME_RE = re.compile(
    r"""
    (?P<B>\d+)_
    (?P<T>\d+)_
    (?P<H>\d+)_
    (?P<W>\d+)
    _rank(?P<rank>\d+)
    _(?P<stage>VAE|DIT)
    (?:_g\d+)?$
    """,
    re.VERBOSE,
)


def parse_task_name(name: str):
    """
    解析 task name:
    1_21_1072_1936_rank4_DIT_g012345
    """
    m = TASK_NAME_RE.search(name)
    if not m:
        raise ValueError(f"Unrecognized task name format: {name}")

    return {
        "B": int(m.group("B")),
        "T": int(m.group("T")),
        "H": int(m.group("H")),
        "W": int(m.group("W")),
        "rank": int(m.group("rank")),
        "stage": m.group("stage"),
    }


# ==============================================================================
# 模块 2: Task Planner（保留，仅用于读取 YAML）
# ==============================================================================

class TaskPlanner:
    def __init__(self, yaml_content):
        if 'tasks' not in yaml_content:
            raise ValueError("YAML must contain a top-level 'tasks' key.")
        self.raw_tasks = yaml_content['tasks']


# ==============================================================================
# 模块 3: FLOPs Simulator（改为 Data-level）
# ==============================================================================

class FlopsSimulator:
    def __init__(self):
        self.vae_config = {
            "in_channels": 33,
            "latent_channels": 16,
            "block_out_channels": (128, 256, 512, 512),
            "layers_per_block": 2,
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8,
            "mid_block_add_attention": True,
            "double_z": True,
        }

        self.dit_config = {
            "hidden_dim": 3072,
            "num_layers": 60,
            "patch_size": (1, 2, 2),
            "in_channels": 16,
            "out_channels": 16,
            "use_checkpointing": True,
        }

    # --------------------------------------------------------------------------
    # Data-level FLOPs
    # --------------------------------------------------------------------------

    def estimate_vae_flops(self, B, T, H, W):
        return estimate_hunyuan_vae_encoder_flops(
            B, T, H, W, **self.vae_config
        )

    def estimate_dit_flops(self, B, T, H, W):
        lt = math.ceil(T / 4)
        lh = math.ceil(H / 8)
        lw = math.ceil(W / 8)
        return estimate_hunyuan_dit_train_flops(
            B, lt, lh, lw, **self.dit_config
        )

    # --------------------------------------------------------------------------
    # Iteration FLOPs
    # --------------------------------------------------------------------------

    def estimate_iteration_flops(self, tasks):
        """
        按 (B,T,H,W,rank) 聚合
        """
        data_map = defaultdict(lambda: {"VAE": False, "DIT": False})

        for task in tasks:
            info = parse_task_name(task["name"])
            key = (
                info["B"],
                info["T"],
                info["H"],
                info["W"],
                info["rank"],
            )
            data_map[key][info["stage"]] = True

        total_flops = 0.0
        per_data_records = []

        for (B, T, H, W, rank), stages in data_map.items():
            flops = 0.0

            if stages["VAE"]:
                flops += self.estimate_vae_flops(B, T, H, W)
            if stages["DIT"]:
                flops += self.estimate_dit_flops(B, T, H, W)

            total_flops += flops

            per_data_records.append({
                "B": B,
                "T": T,
                "H": H,
                "W": W,
                "rank": rank,
                "has_vae": stages["VAE"],
                "has_dit": stages["DIT"],
                "flops": flops,
            })

        return total_flops, per_data_records


# ==============================================================================
# 模块 4: Experiment Analyzer（Iteration-level MFU）
# ==============================================================================

BASE_SCHEDULE_PATH = r"t2v_flow/planner/generated_schedules"
BASE_LOG_PATH = r"dynamic_flex_exp_log/arachne"

GPU_PEAK_TFLOPS = 989.0


class ExperimentAnalyzer:
    def __init__(self, model_str, resolution, schedule_type):
        self.model_str = model_str
        self.resolution = resolution
        self.schedule_type = schedule_type

        if "hunyuan" in model_str:
            schedule_str = "hunyuan"
        else:
            raise ValueError("Unknown model string")

        self.schedule_dir = os.path.join(
            BASE_SCHEDULE_PATH, schedule_str, resolution, schedule_type
        )

        self.time_log_path = os.path.join(
            BASE_LOG_PATH, model_str, resolution,
            "figures", "iteration_times_summary.json"
        )

        self.simulator = FlopsSimulator()

    def run(self):
        print(f"[Analyzer] {self.model_str} / {self.resolution} / {self.schedule_type}")

        if not os.path.exists(self.time_log_path):
            print(f"[Error] Missing time log: {self.time_log_path}")
            return None

        with open(self.time_log_path, "r") as f:
            time_data = json.load(f)

        yaml_files = sorted(glob.glob(
            os.path.join(self.schedule_dir, "schedule_*.yaml")
        ))

        results = []

        for y_path in yaml_files:
            m = re.search(r"schedule_(\d+)\.yaml", os.path.basename(y_path))
            if not m:
                continue

            iter_id = m.group(1)
            if iter_id not in time_data:
                continue

            exec_time = float(time_data[iter_id])
            if exec_time <= 0:
                continue

            with open(y_path, "r") as f:
                plan_data = yaml.safe_load(f)

            planner = TaskPlanner(plan_data)

            total_flops, per_data = self.simulator.estimate_iteration_flops(
                planner.raw_tasks
            )

            num_gpus = max(
                max(task["gpus"]) for task in planner.raw_tasks
            ) + 1

            peak_flops = num_gpus * GPU_PEAK_TFLOPS * 1e12 * exec_time
            mfu = total_flops / peak_flops

            results.append({
                "Iteration": int(iter_id),
                "Time(s)": exec_time,
                "GPUs": num_gpus,
                "Num Data": len(per_data),
                "Total FLOPs (T)": total_flops / 1e12,
                "Cluster MFU": mfu,
            })

            print(
                f"[Iter {iter_id}] "
                f"Time={exec_time:.2f}s | "
                f"Data={len(per_data)} | "
                f"MFU={mfu*100:.2f}%"
            )

        return pd.DataFrame(results).sort_values("Iteration")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hunyuan-129")
    parser.add_argument("--res", type=str, default="720p")
    parser.add_argument("--schedule", type=str, default="flex_sp")
    parser.add_argument("--output", type=str, default="mfu_summary.csv")
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.model, args.res, args.schedule)
    df = analyzer.run()

    if df is not None and not df.empty:
        print("\n" + "=" * 80)
        print(df.to_string(index=False, formatters={
            "Cluster MFU": "{:.2%}".format,
            "Time(s)": "{:.3f}".format,
        }))
        df.to_csv(args.output, index=False)
        print(f"\n[Saved] {args.output}")
    else:
        print("[Error] No valid data processed.")
