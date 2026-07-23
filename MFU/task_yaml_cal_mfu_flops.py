"""
MFU and FLOPs analysis for distributed schedule YAMLs.

Purpose
- Parse schedule_*.yaml tasks and simulate per-rank FLOPs for VAE and DiT stages.
- Combine simulated FLOPs with measured iteration time to report MFU metrics.

FLOPs model references
- VAE: MFU/calculate_vae_mfu.py::estimate_hunyuan_vae_encoder_flops
  - 3D conv FLOPs: 2 * B * T * H * W * Cin * Cout * K^3
  - Adds stem, down blocks (resnet + downsample), mid attention, and final conv.
  - This script tiles the input and sums per-tile FLOPs across ranks.
- DiT: MFU/calculate_dit_mfu.py::estimate_hunyuan_dit_train_flops
  - Global sequence length: s = (T/pt)*(H/ph)*(W/pw) + text_length
  - Per-layer FLOPs: (8 + 4*mlp_ratio) * B * s * h^2 + 4 * B * s^2 * h
  - Total train FLOPs = (IO + layers) / sp_size * (4 if checkpointing else 3)
  - This script divides the global train FLOPs evenly across SP ranks.

Task conventions
- Task name encodes shape as "bs_t_h_w"; otherwise defaults to 1x1x256x256.
- task_type must be "VAE" or "DIT"; others contribute 0 FLOPs.

I/O paths (defaults)
- schedules: t2v_flow/planner/generated_schedules/<model>/<resolution>/<schedule_type>
- time logs: <base_log_dir>/<framework>/<model>/<resolution>/figures/iteration_times_summary.json
- outputs: <base_log_dir>/<framework>/<model>/<resolution>/figures/flops/

CLI example
python MFU/task_yaml_cal_mfu_flops.py --model wan1.3b --resolution 720p   --schedule-type megatron-lm --base-log-dir baseline_log
"""
# ==============================================================================
import yaml
import math
import numpy as np
from collections import defaultdict, deque
from .calculate_dit_mfu import estimate_hunyuan_dit_train_flops
from .calculate_vae_mfu import estimate_hunyuan_vae_encoder_flops
import argparse
import os
import sys
from collections import defaultdict, deque
import json, glob, re
import pandas as pd
# ==============================================================================
# ==============================================================================
class TaskPlanner:
    def __init__(self, yaml_content):
        if 'tasks' not in yaml_content:
            raise ValueError("YAML must contain a top-level 'tasks' key.")
        self.raw_tasks = yaml_content['tasks']
        self.task_map = {t['name']: t for t in self.raw_tasks}
        
    def generate_per_rank_plans(self):
        global_sorted_tasks = self._topological_sort()

        print(f"[Planner] Topological sort completed. Total tasks: {global_sorted_tasks}")
        rank_queues = defaultdict(list)
        
        print(f"[Planner] Generated global order with {len(global_sorted_tasks)} tasks.")
        
        for task in global_sorted_tasks:
            assigned_gpus = task['gpus']
            for gpu_id in assigned_gpus:
                rank_queues[gpu_id].append(task)
                
        return rank_queues

    def _topological_sort(self):
        in_degree = {t['name']: 0 for t in self.raw_tasks}
        graph = defaultdict(list)
        
        for t in self.raw_tasks:
            name = t['name']
            deps = t.get('dependencies', [])
            for dep in deps:
                if dep in self.task_map:
                    graph[dep].append(name)
                    in_degree[name] += 1
        
        queue = deque([n for n in in_degree if in_degree[n] == 0])
        sorted_result = []
        
        while queue:
            u_name = queue.popleft()
            sorted_result.append(self.task_map[u_name])
            for v_name in graph[u_name]:
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0:
                    queue.append(v_name)
                    
        if len(sorted_result) != len(self.raw_tasks):
            raise ValueError("Cycle detected in task dependencies!")
        return sorted_result

# ==============================================================================
# ==============================================================================
class FlopsSimulator:
    def __init__(self):
        self.vae_config = {
            "in_channels": 33, "latent_channels": 16,
            "block_out_channels": (128, 256, 512, 512), "layers_per_block": 2,
            "temporal_compression_ratio": 4, "spatial_compression_ratio": 8,
            "mid_block_add_attention": True, "double_z": True
        }
        self.vae_tile_config = {
            "tile_h": 256, "h_stride": 192,
            "tile_w": 256, "w_stride": 192,
            "tile_frame": 16, "frame_stride": 12
        }
        self.dit_config = {
            "hidden_dim": 3072, "num_layers": 60, "patch_size": (1, 2, 2),
            "in_channels": 16, "out_channels": 16, "use_checkpointing": True,
            "text_length": 256
        }
        self._task_simulation_cache = {}

    def parse_task_dims(self, name):
        try:
            p = name.split('_')
            return int(p[0]), int(p[1]), int(p[2]), int(p[3])
        except:
            return 1, 1, 256, 256

    def evaluate_rank_flops(self, global_rank_id, task_list):
        """
        Returns: (total_flops, detailed_task_list)
        """
        total_flops = 0.0
        detailed_records = []
        
        for task in task_list:
            name = task['name']
            task_type = task['args']['task_type']
            assigned_gpus = task['gpus']
            print(f"Evaluating Task: {name} | Type: {task_type} | GPUs: {assigned_gpus}")
            try:
                local_rank_index = assigned_gpus.index(global_rank_id)
            except ValueError:
                continue 
                
            sp_degree = len(assigned_gpus)
            bs, t, h, w = self.parse_task_dims(name)
            
            if name not in self._task_simulation_cache:
                if task_type == 'VAE':
                    dist = self._simulate_vae_distribution(bs, t, h, w, sp_degree)
                elif task_type == 'DIT':
                    dist = self._simulate_dit_distribution(bs, t, h, w, sp_degree)
                else:
                    dist = defaultdict(float)
                self._task_simulation_cache[name] = dist
            
            load_distribution = self._task_simulation_cache[name]
            task_flops = load_distribution[local_rank_index]
            
            detailed_records.append({
                'name': name,
                'type': task_type,
                'shape': f"{bs}x{t}x{h}x{w}",
                'sp': sp_degree,
                'flops': task_flops
            })
            
            total_flops += task_flops
            
        return total_flops, detailed_records

    def _simulate_vae_distribution(self, bs, t, h, w, sp_degree):
        rank_flops = defaultdict(float)
        for r in range(sp_degree): rank_flops[r] = 0.0
        tc = self.vae_tile_config
        
        for f_start in range(0, t, tc['frame_stride']):
            f_end = min(f_start + tc['tile_frame'] + 1, t)
            chunk_t = f_end - f_start
            
            spatial_coords = []
            for h_s in range(0, h, tc['h_stride']):
                for w_s in range(0, w, tc['w_stride']):
                    spatial_coords.append((h_s, w_s))
            
            tiles_by_shape = defaultdict(list)
            for (h_s, w_s) in spatial_coords:
                h_e = min(h_s + tc['tile_h'], h)
                w_e = min(w_s + tc['tile_w'], w)
                shape = (chunk_t, h_e - h_s, w_e - w_s)
                tiles_by_shape[shape].append(shape)
                
            for shape, tiles in sorted(tiles_by_shape.items()):
                count = len(tiles)
                for r in range(sp_degree):
                    num = len(range(r, count, sp_degree))
                    if num > 0:
                        f = estimate_hunyuan_vae_encoder_flops(
                            bs, shape[0], shape[1], shape[2], **self.vae_config
                        )
                        rank_flops[r] += f * num
        return rank_flops

    def _simulate_dit_distribution(self, bs, t, h, w, sp_degree):
        lt = math.ceil(t / 4); lh = math.ceil(h / 8); lw = math.ceil(w / 8)
        global_f = estimate_hunyuan_dit_train_flops(bs, lt, lh, lw, **self.dit_config)
        per_rank = global_f / sp_degree
        return {r: per_rank for r in range(sp_degree)}

# ==============================================================================
# ==============================================================================
def export_detailed_report(filepath, results_map):
    """
    results_map: {global_rank: (total_flops, [task_details])}
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Distributed FLOPS Analysis Report\n")
            f.write(f"==================================================\n\n")
            
            sorted_ranks = sorted(results_map.keys())
            
            all_totals = [res[0] for res in results_map.values()]
            if all_totals:
                max_f = max(all_totals)
                avg_f = sum(all_totals) / len(all_totals)
                f.write(f"Cluster Summary:\n")
                f.write(f"  Max Load (Bottleneck): {max_f/1e12:.4f} TFLOPs\n")
                f.write(f"  Avg Load:              {avg_f/1e12:.4f} TFLOPs\n")
                f.write(f"  Imbalance Ratio:       {max_f/min(all_totals):.4f}\n\n")
                f.write(f"{'='*50}\n\n")

            for rank in sorted_ranks:
                total_flops, tasks = results_map[rank]
                
                f.write(f"Global Rank {rank:02d}\n")
                f.write(f"Total Load: {total_flops/1e12:.4f} TFLOPs\n")
                f.write(f"Task Breakdown:\n")
                f.write(f"  {'Task Name':<45} | {'Type':<4} | {'Shape':<12} | {'SP':<3} | {'FLOPS (T)'}\n")
                f.write(f"  {'-'*45} | {'-'*4} | {'-'*12} | {'-'*3} | {'-'*9}\n")
                
                for task in tasks:
                    flops_t = task['flops'] / 1e12
                    f.write(f"  {task['name']:<45} | {task['type']:<4} | {task['shape']:<12} | {task['sp']:<3} | {flops_t:.4f}\n")
                
                f.write(f"\n{'-'*80}\n\n")
                
        print(f"\n[Success] Detailed report written to: {filepath}")
        
    except Exception as e:
        print(f"[Error] Failed to write report: {e}")

# ==============================================================================
# ==============================================================================


BASE_SCHEDULE_PATH = r"t2v_flow/planner/generated_schedules" 
BASE_LOG_PATH = r"dynamic_flex_exp_log/arachne" 

# H100 Spec
GPU_PEAK_TFLOPS = 989.0

class ExperimentAnalyzer:
    def __init__(self, model_str, resolution, schedule_type,
                 base_log_dir="dynamic_flex_exp_log", max_frames="",
                framework=None, plan_max_frames=None,):
        self.model_str = model_str
        self.resolution = resolution
        self.schedule_type = schedule_type


        self.model_str = model_str
        self.resolution = resolution
        self.schedule_type = schedule_type

        self.framework = framework or schedule_type
        self.base_log_dir = base_log_dir
        self.max_frames = max_frames
        self.plan_max_frames = max_frames if plan_max_frames is None else plan_max_frames

        if "hunyuan" in self.model_str:
            schedule_model_str = "hunyuan"
        self.schedule_dir = os.path.join(
            BASE_SCHEDULE_PATH,
            schedule_model_str,
            self.resolution,
            self.plan_max_frames,
            self.schedule_type,
        )

        # dynamic_flex_exp_log\arachne\hunyuan-129\720p\figures\iteration_times_summary.json
        self.time_log_path = os.path.join(
            self.base_log_dir,
            self.framework,
            self.model_str,
            self.resolution,
            self.max_frames,
            "figures",
            "iteration_times_summary.json",
        )


        self.flops_output_dir = os.path.join(
            self.base_log_dir,
            self.framework,
            self.model_str,
            self.resolution,
            self.max_frames,
            "figures",
            "flops",
        )
        os.makedirs(self.flops_output_dir, exist_ok=True)
        
        self.simulator = FlopsSimulator()

    def run(self):
        print(f"Analyzer Started for: {self.model_str} / {self.resolution} / {self.schedule_type}")
        print(f"  > Schedule Dir: {self.schedule_dir}")
        print(f"  > Time Log:     {self.time_log_path}")

        if not os.path.exists(self.time_log_path):
            print(f"[Error] Time log not found: {self.time_log_path}")
            return None
        
        with open(self.time_log_path, 'r', encoding='utf-8-sig') as f:
            time_data = json.load(f) # {"0": 1.5, "1": 1.4 ...}
        
        yaml_pattern = os.path.join(self.schedule_dir, "schedule_*.yaml")
        yaml_files = glob.glob(yaml_pattern)
        
        if not yaml_files:
            print(f"[Warning] No schedule YAML files found in {self.schedule_dir}")
            return None

        results = []

        for y_path in sorted(yaml_files):
            filename = os.path.basename(y_path)
            match = re.search(r'schedule_(\d+)\.yaml', filename)
            if not match: continue
            
            iter_id = match.group(1)
            
            if iter_id not in time_data:
                print(f"[Skip] No time data for Iteration {iter_id}")
                continue
            
            exec_time = float(time_data[iter_id])
            if exec_time <= 0: continue

            with open(y_path, 'r') as f:
                plan_data = yaml.safe_load(f)
            print(f"\n[Iteration {iter_id}] Loaded schedule YAML: {y_path}")
            planner = TaskPlanner(plan_data)
            rank_queues = planner.generate_per_rank_plans()
            
            rank_flops_map = {}          # {rank: total_flops}
            rank_task_details = {}      # {rank: [task_details]}

            all_gpus = sorted(rank_queues.keys())
            print(f"  [Iter {iter_id}] Simulating FLOPs for {all_gpus} GPUs... rank_queues keys: {list(rank_queues.keys())}")
            for gpu in all_gpus:
                total_flops, detailed_tasks = self.simulator.evaluate_rank_flops(
                    gpu, rank_queues[gpu]
                )
                rank_flops_map[gpu] = total_flops
                rank_task_details[gpu] = detailed_tasks

            rank_flops_list = list(rank_flops_map.values())
            if not rank_flops_list:
                continue


            cluster_total_flops = sum(rank_flops_list)
            max_rank_flops = max(rank_flops_list)
            min_rank_flops = min(rank_flops_list)
            num_gpus = len(rank_flops_list)

            cluster_peak = num_gpus * GPU_PEAK_TFLOPS * 1e12 * exec_time
            single_peak = GPU_PEAK_TFLOPS * 1e12 * exec_time
            
            cluster_mfu = cluster_total_flops / cluster_peak
            bottleneck_mfu = max_rank_flops / single_peak
            
            results.append({
                "Iteration": int(iter_id),
                "Time(s)": exec_time,
                "GPUs": num_gpus,
                "Max Rank Load (T)": max_rank_flops / 1e12,
                "Cluster Total (T)": cluster_total_flops / 1e12,
                "Cluster MFU": cluster_mfu,
                "Bottleneck MFU": bottleneck_mfu,
                "Imbalance Penalty": bottleneck_mfu - cluster_mfu
            })
            
            print(f"  [Iter {iter_id}] Time: {exec_time:.2f}s | Cluster MFU: {cluster_mfu*100:.2f}% | Bottleneck: {bottleneck_mfu*100:.2f}%")


            # =========================
            # =========================
            iter_flops_record = {
                "iteration": int(iter_id),
                "time_s": exec_time,
                "num_gpus": len(rank_flops_map),
                "per_rank_flops": {
                    str(rank): {
                        "total_flops": flops,
                        "total_tflops": flops / 1e12,
                        "tasks": rank_task_details[rank],
                    }
                    for rank, flops in rank_flops_map.items()
                },
                "summary": {
                    "cluster_total_flops": sum(rank_flops_map.values()),
                    "cluster_total_tflops": sum(rank_flops_map.values()) / 1e12,
                    "max_rank_flops": max(rank_flops_map.values()),
                    "min_rank_flops": min(rank_flops_map.values()),
                    "imbalance_ratio": max(rank_flops_map.values()) / min(rank_flops_map.values()),
                },
            }

            flops_output_path = os.path.join(
                self.flops_output_dir,
                f"iteration_{int(iter_id):03d}_flops.json",
            )

            with open(flops_output_path, "w") as f:
                json.dump(iter_flops_record, f, indent=2)

            print(f"    [FLOPs] Per-rank FLOPs written to: {flops_output_path}")


            # =======================================================
            # =======================================================
            
            flops_vals = np.array(list(rank_flops_map.values()))
            f_avg = np.mean(flops_vals)
            f_max = np.max(flops_vals)
            f_min = np.min(flops_vals)
            f_std = np.std(flops_vals)
            
            if f_avg > 0:
                overhead = (f_max - f_avg) / f_avg
                cv = f_std / f_avg
            else:
                overhead = 0.0
                cv = 0.0
            
            if f_min > 0:
                max_min_ratio = f_max / f_min
            else:
                max_min_ratio = 0.0

            simple_data_with_metrics = {
                "metrics": {
                    "imbalance_overhead": overhead,
                    "max_min_ratio": max_min_ratio,      # Max/Min
                    "coefficient_of_variation": cv,      # Std/Avg
                    "cluster_total_flops": float(np.sum(flops_vals))
                },
                "ranks": {str(k): v for k, v in rank_flops_map.items()}
            }
            
            simple_output_path = os.path.join(
                self.flops_output_dir,
                f"iteration_{int(iter_id):03d}_flops_simple.json",
            )

            with open(simple_output_path, "w") as f:
                json.dump(simple_data_with_metrics, f, indent=2)
            
            print(f"    [FLOPs] Simple (w/ Metrics) written to: {simple_output_path}")
        return pd.DataFrame(results).sort_values("Iteration")

# ==============================================================================
# ==============================================================================
if __name__ == "__main__":

    # =======================
    # =======================
    parser = argparse.ArgumentParser(
        description="Evaluate generated schedule_*.yaml plans to obtain per-rank FLOPs "
                    "(iteration_XXX_flops_simple.json) and an MFU summary.")
    parser.add_argument("--model", default="hunyuan-129",
                        help="Model tag used in the log path, e.g. hunyuan-129 / wan1.3b / cogvideox.")
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--max-frames", dest="max_frames", default="",
                        help="Frame-window level in the LOG path, e.g. '129'; '' if the "
                             "model tag already encodes it (e.g. hunyuan-129).")
    parser.add_argument("--plan-max-frames", dest="plan_max_frames", default=None,
                        help="Frame-window level in the SCHEDULE path, e.g. '129' "
                             "(plans live at .../<res>/<frames>/<sched>). "
                             "Defaults to --max-frames.")
    parser.add_argument("--schedule-type", dest="schedule_type", default="flex_sp",
                        help="genetic | megatron-lm | flex_sp")
    parser.add_argument("--framework", default=None,
                        help="Log framework dir; defaults from --schedule-type "
                             "(genetic -> arachne, otherwise same).")
    parser.add_argument("--base-log-dir", dest="base_log_dir", default="baseline_log")
    parser.add_argument("--output-csv", dest="output_csv", default="mfu_summary.csv")
    args = parser.parse_args()

    model = args.model
    resolution = args.resolution
    schedule_type = args.schedule_type
    max_frames = args.max_frames
    base_log_dir = args.base_log_dir
    output_csv = args.output_csv
    if args.framework:
        framework = args.framework
    elif schedule_type == "genetic":
        framework = "arachne"
    else:
        framework = schedule_type

    # =======================
    # =======================
    analyzer = ExperimentAnalyzer(
        model_str=model,
        resolution=resolution,
        schedule_type=schedule_type,
        base_log_dir=base_log_dir,
        framework=framework,
        max_frames=max_frames,
        plan_max_frames=args.plan_max_frames,
    )

    # =======================
    # =======================
    df = analyzer.run()

    if df is not None and not df.empty:
        print("\n" + "=" * 80)
        print(f"SUMMARY REPORT ({model}/{resolution}/{schedule_type})")
        print("=" * 80)

        print(df.to_string(index=False, formatters={
            'Cluster MFU': '{:.2%}'.format,
            'Bottleneck MFU': '{:.2%}'.format,
            'Imbalance Penalty': '{:.2%}'.format,
            'Time(s)': '{:.3f}'.format,
            'Max Rank Load (T)': '{:.3f}'.format
        }))

        df.to_csv(output_csv, index=False)
        print(f"\n[Success] Report saved to {output_csv}")
    else:
        print("\n[Error] No valid data processed.")
