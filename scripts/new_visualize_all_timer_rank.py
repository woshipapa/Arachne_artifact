"""
Workflow
1) Set model/resolution/framework/base_log_dir in __main__.
2) Ensure logs exist at: {base_log_dir}/{framework}/{model}/{resolution}/timing_rank*.log
3) Run this script to parse logs, compute averages, and generate outputs.
4) Inspect JSON summaries and PNG timelines in the figures output directory.

Usage
- Default: run `python dynamic_flex_exp_log/new_visualize_all_timer_rank.py`
- Toggle timing mode with USE_PARENT_ITERATION_TIME:
  * True: use parent "iteration-X" CUDA time (includes overhead).
  * False: sum child stages (excludes overhead).

What the code does
- parse_and_group_runs: scans timing_rank*.log, groups stage timings by true iteration
  and rank, and guards against reset/backtrack in iteration ids.
- analyze_and_average_runs: removes outliers via IQR, averages stage times, derives
  per-rank active/idle time, and computes per-iteration makespan.
- plot_iteration_gantt_chart: renders per-rank timelines with idle/overhead slices.
- __main__: wires everything together, writes JSON summaries, and saves figures.
"""
import os
import re
import glob
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


# ================= 配置开关 =================
# True: 使用日志中 'iteration-X' 父节点的 CUDA 时间作为总耗时 (包含warmup和steady之后的平均)
# False: 使用子阶段 (VAE+DiT...) 平均耗时的【累加和】作为总耗时 
USE_PARENT_ITERATION_TIME = False
# ===========================================

def format_stage_name(original_name):
    """Parses specific stage names into a more readable format."""
    dit_forward_pattern = re.match(r'forward_single\d+_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp(\d+)', original_name)
    if dit_forward_pattern:
        bs, f, h, w, sp = dit_forward_pattern.groups()
        return f"DiT fwd (bs={bs}, f={f}, h={h}, w={w}, SP={sp})"
    dit_backward_pattern = re.match(r'backward_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp_(\d+)', original_name)
    if dit_backward_pattern:
        bs, f, h, w, sp = dit_backward_pattern.groups()
        return f"DiT bwd (bs={bs}, f={f}, h={h}, w={w}, SP={sp})"
    vae_forward_pattern = re.match(r'vae_forward_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp_(\d+)_dist(.*)', original_name)
    if vae_forward_pattern:
        bs, f, h, w, sp, dist_tag = vae_forward_pattern.groups()
        dist_tag = dist_tag if dist_tag else "N/A"
        return f"VAE fwd (bs={bs}, f={f}, h={h}, w={w}, SP={sp}, dist={dist_tag})"
    return original_name

def parse_and_group_runs(log_search_path):
    """
    解析日志。
    包含【单调递增过滤】逻辑：
    如果遇到 new_id < max_iter，说明发生了 Reset 且处于重跑旧数据阶段 -> 【丢弃】。
    """
    all_runs_data = defaultdict(lambda: defaultdict(list))
    # 存储父阶段(iteration-X)的原始总耗时 [true_iter][rank] -> [time1, time2, ...]
    iteration_raw_durations = defaultdict(lambda: defaultdict(list))
    
    # 记录每个 Rank 见过的最大 Iteration ID，初始为 -1
    max_iter_per_rank = defaultdict(lambda: -1)
    
    log_files = glob.glob(log_search_path)
    
    if not log_files:
        print(f"Warning: No log files found matching '{log_search_path}'.")
        return None, None

    line_parser = re.compile(
        r"\[.*?\]\s+\[Rank\s+(\d+)\]\s+.*?(?:L\s+)?Stage\s+'(?P<name>.*?)'(?:\s+\[.*?\])?:\s+CPU\s+[\d.]+\s*ms,\s+CUDA\s+(?P<cuda>[\d.]+)\s*ms"
    )

    for log_file in sorted(log_files):
        print(f"Parsing {os.path.basename(log_file)}...")
        
        current_true_iter_id = None
        current_rank = -1
        current_run_buffer = []
        
        def flush_buffer():
            nonlocal current_run_buffer
            if current_true_iter_id is not None and current_run_buffer:
                all_runs_data[current_true_iter_id][current_rank].append(current_run_buffer)
            current_run_buffer = []

        with open(log_file, 'r') as f:
            for line in f:
                match = line_parser.search(line)
                if match:
                    rank = int(match.group(1))
                    stage_name = match.group('name')
                    cuda_time_s = float(match.group('cuda')) / 1000.0
                    
                    if rank != current_rank:
                        flush_buffer()
                        current_rank = rank

                    # --- 核心判断逻辑 ---
                    
                    if stage_name.startswith('iteration-'):
                        flush_buffer()
                        try:
                            new_id = int(stage_name.split('-')[-1])
                            
                            # [回退过滤逻辑]
                            if new_id < max_iter_per_rank[rank]:
                                # 现在的 ID 比历史最大值小，丢弃
                                current_true_iter_id = None
                            else:
                                # 现在的 ID >= 历史最大值，保留
                                max_iter_per_rank[rank] = new_id
                                current_true_iter_id = new_id
                                # 记录父阶段 CUDA 时间
                                iteration_raw_durations[new_id][rank].append(cuda_time_s)
                                
                        except ValueError:
                            print(f"Skipping malformed iteration name: {stage_name}")
                            current_true_iter_id = None

                    else:
                        # 子节点：只有在 current_true_iter_id 有效时才收集
                        if current_true_iter_id is not None:
                            current_run_buffer.append({
                                'name': stage_name, 
                                'duration_s': cuda_time_s
                            })

        flush_buffer()

    return all_runs_data, iteration_raw_durations

def remove_outliers_iqr(data):
    """Removes outliers from a list of numbers using the IQR method."""
    if len(data) < 4: return data, []
    data = np.array(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return clean_data.tolist(), outliers.tolist()

def analyze_and_average_runs(all_runs_data, iteration_raw_durations, use_parent_time=True):
    """
    修改点：
    根据 use_parent_time 开关决定 rank_total_times 的计算方式。
    """
    analysis_results = defaultdict(lambda: defaultdict(dict))
    averaged_data = defaultdict(dict)

    for true_iter, ranks_data in all_runs_data.items():
        # First pass: Calculate averaged data for child stages
        for rank, runs in ranks_data.items():
            if not runs: continue
            
            num_stages = len(runs[0]) 
            stage_analysis_for_json = []
            averaged_stages_for_plot = []

            for i in range(num_stages):
                if i >= len(runs[0]): break 
                stage_name = runs[0][i]['name']
                original_durations = [run[i]['duration_s'] for run in runs if len(run) > i]
                cleaned_durations, outliers = remove_outliers_iqr(original_durations)
                avg_duration = np.mean(cleaned_durations) if cleaned_durations else (np.mean(original_durations) if original_durations else 0)
                
                stage_analysis_for_json.append({
                    "stage_name": format_stage_name(stage_name),
                    "original_durations_s": original_durations,
                    "cleaned_durations_s": cleaned_durations,
                    "removed_outliers": outliers,
                    "average_duration_s": round(avg_duration, 4)
                })
                averaged_stages_for_plot.append({'name': stage_name, 'duration_s': avg_duration})
            
            analysis_results[f"iteration_{true_iter}"][f"rank_{rank}"] = {
                "total_runs_found": len(runs),
                "stage_analysis": stage_analysis_for_json
            }
            averaged_data[true_iter][rank] = averaged_stages_for_plot
        
        # Second pass: Calculate total times based on the switch
        if true_iter in averaged_data:
            # 1. 计算 Active Time (子阶段平均值之和) - 始终计算
            rank_active_times = {rank: sum(s['duration_s'] for s in stages) for rank, stages in averaged_data[true_iter].items()}
            
            # 2. 计算 Total Time (根据开关)
            rank_final_total_times = {}
            
            for rank in averaged_data[true_iter].keys():
                if use_parent_time:
                    # 模式A：使用父节点 'iteration-X' 的记录 (包含 overhead)
                    raw_times = iteration_raw_durations[true_iter][rank]
                    if raw_times:
                        rank_final_total_times[rank] = np.mean(raw_times)
                    else:
                        rank_final_total_times[rank] = rank_active_times.get(rank, 0.0)
                else:
                    # 模式B：使用子阶段累加和 (不含 overhead)
                    rank_final_total_times[rank] = rank_active_times.get(rank, 0.0)

            # 3. 处理 DiT 形状统计 (不变)
            for rank, stages in averaged_data[true_iter].items():
                dit_times_by_shape = defaultdict(float)
                for stage in stages:
                    if stage['name'].startswith('forward_single') or stage['name'].startswith('backward_bs'):
                        formatted_name = format_stage_name(stage['name'])
                        dit_times_by_shape[formatted_name] += stage['duration_s']
                
                dit_times_by_shape_rounded = {name: round(time, 4) for name, time in dit_times_by_shape.items()}
                if dit_times_by_shape_rounded:
                    analysis_results[f"iteration_{true_iter}"][f"rank_{rank}"]["dit_times_by_shape"] = dit_times_by_shape_rounded

            # 4. 计算 Makespan 和 Idle
            if rank_final_total_times:
                iteration_makespan = max(rank_final_total_times.values())
                analysis_results[f"iteration_{true_iter}"]["iteration_total_time_s"] = round(iteration_makespan, 4)

                for rank in rank_active_times.keys():
                    active_time = rank_active_times[rank]
                    final_total_time = rank_final_total_times[rank]
                    
                    # Idle time = Final Total - Active
                    # 如果 USE_PARENT_ITERATION_TIME = False，这里 idle 理论上为 0
                    idle_time = max(0, final_total_time - active_time)
                    
                    rank_key = f"rank_{rank}"
                    analysis_results[f"iteration_{true_iter}"][rank_key]["active_time_s"] = round(active_time, 4)
                    analysis_results[f"iteration_{true_iter}"][rank_key]["idle_time_s"] = round(idle_time, 4)
                    analysis_results[f"iteration_{true_iter}"][rank_key]["calc_mode"] = "Parent Log" if use_parent_time else "Sum of Children"

    return analysis_results, averaged_data

def plot_iteration_gantt_chart(iteration_num, iteration_data, analysis_result_for_iter, output_dir):
    fig, ax = plt.subplots(figsize=(20, 10))
    ranks = sorted(iteration_data.keys())
    
    all_stage_names = {format_stage_name(stage['name']) for rank_data in iteration_data.values() for stage in rank_data}
    all_stage_names.add("Idle/Overhead")
    
    colors = cm.get_cmap('tab20', len(all_stage_names))
    color_map = {name: colors(i) for i, name in enumerate(all_stage_names)}
    color_map["Idle/Overhead"] = '#d3d3d3' 

    global_makespan = analysis_result_for_iter.get("iteration_total_time_s", 0)

    for rank in ranks:
        current_time = 0.0
        for stage in iteration_data.get(rank, []):
            start_time, duration = current_time, stage['duration_s']
            formatted_name = format_stage_name(stage['name'])
            ax.barh(f"Rank {rank}", duration, left=start_time, height=0.6,
                    edgecolor='black', color=color_map[formatted_name], align='center')
            current_time += duration
        
        # 只有当总时间包含 overhead 时，这个灰色条才会出现
        idle_time = global_makespan - current_time
        if idle_time > 1e-4:
            ax.barh(f"Rank {rank}", idle_time, left=current_time, height=0.6,
                    edgecolor='grey', color=color_map["Idle/Overhead"], hatch='..', align='center')

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title(f"Averaged Timeline for True Iteration {iteration_num} (Makespan: {global_makespan}s)", fontsize=16, weight='bold')
    
    legend_patches = [mpatches.Patch(color=color, label=name) for name, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'averaged_true_iteration_{iteration_num}_timeline.png')
    plt.savefig(output_filename, dpi=150)
    print(f"Chart saved to: {output_filename}")
    plt.close(fig)

def create_dummy_logs():
    print("Creating dummy log files for demonstration...")
    # 模拟数据包含 iteration-X，方便测试两种模式
    log_content = """[2025-12-15 22:28:25] [Rank 0] [INFO] - [Iter 11] Stage 'iteration-2' [id=45]: CPU 37963.769ms, CUDA 37963.746ms
[2025-12-15 22:28:25] [Rank 0] [INFO] - [Iter 11]   L Stage 'vae_forward_bs_1_f_97_h_1280_w_720_sp_4_distNone' [id=46]: CPU 5237.175ms, CUDA 5237.188ms
[2025-12-15 22:28:25] [Rank 0] [INFO] - [Iter 11]   L Stage 'forward_single20_bs_1_f_97_h_1280_w_720_sp4' [id=47]: CPU 7555.521ms, CUDA 7795.943ms
[2025-12-15 22:28:25] [Rank 0] [INFO] - [Iter 11]   L Stage 'backward_bs_1_f_97_h_1280_w_720_sp_4' [id=48]: CPU 19943.338ms, CUDA 20628.496ms
"""
    with open("timer_rank0.log", "w") as f: f.write(log_content)

if __name__ == "__main__":
    model, resolution = "hunyuan", "720p"
    # base_log_dir = "baseline_log"
    base_log_dir = "dynamic_flex_exp_log"
    # framework = "flex_sp"
    framework = "megatron-lm"
    framework = "arachne"
    partial = "105"
    log_search_path = os.path.join(base_log_dir,framework ,model, resolution, partial, "timing_rank*.log")
    output_figure_dir = os.path.join(base_log_dir, framework ,model, resolution, partial, "figures")

    print(f"Searching logs in: {log_search_path}")
    print(f"Calculation Mode: {'Using Parent Log Time (Including Overhead)' if USE_PARENT_ITERATION_TIME else 'Using Sum of Children (Excluding Overhead)'}")
    
    all_runs_data, iteration_raw_durations = parse_and_group_runs(log_search_path)

    if all_runs_data is None:
        create_dummy_logs()
        all_runs_data, iteration_raw_durations = parse_and_group_runs("timing_rank*.log")
        output_figure_dir = "figures"

    if all_runs_data:
        # 传入 USE_PARENT_ITERATION_TIME 开关
        analysis_results, averaged_data = analyze_and_average_runs(all_runs_data, iteration_raw_durations, use_parent_time=USE_PARENT_ITERATION_TIME)
        
        os.makedirs(output_figure_dir, exist_ok=True)
        json_output_path = os.path.join(output_figure_dir, 'analysis_summary.json')
        with open(json_output_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        print(f"\nDetailed analysis saved to: {json_output_path}")
        
        final_iteration_times = {}
        for iter_key, iter_data in sorted(analysis_results.items()):
            numeric_iter = int(iter_key.split('_')[-1])
            if "iteration_total_time_s" in iter_data:
                final_iteration_times[numeric_iter] = iter_data["iteration_total_time_s"]
        
        if final_iteration_times:
            times_summary_path = os.path.join(output_figure_dir, 'iteration_times_summary.json')
            with open(times_summary_path, 'w') as f:
                json.dump(final_iteration_times, f, indent=4, sort_keys=True)
            print(f"Final iteration times summary saved to: {times_summary_path}")

        print(f"\nFigures will be saved to: {os.path.abspath(output_figure_dir)}")
        for iter_num, iter_data in sorted(averaged_data.items()):
            iter_key = f"iteration_{iter_num}"
            if iter_key in analysis_results:
                plot_iteration_gantt_chart(iter_num, iter_data, analysis_results[iter_key], output_figure_dir)
        print("\nAll charts generated successfully!")
    else:
        print("Could not process any log data.")