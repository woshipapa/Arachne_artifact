import os
import re
import glob
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

def format_stage_name(original_name):
    """Parses specific stage names into a more readable format."""
    # This function remains unchanged
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
    Parses all logs, grouping runs and handling training resets.
    """
    # This function remains unchanged
    all_runs_data = defaultdict(lambda: defaultdict(list))
    log_files = glob.glob(log_search_path)
    
    if not log_files:
        print(f"Warning: No log files found matching '{log_search_path}'.")
        return None

    line_parser = re.compile(
        r"\[.*?\]\s+\[Rank\s+(\d+)\]\s+.*\[Iter\s+\d+\]\s+Stage\s+'(.*?)':\s+CPU\s+[\d.]+\s*ms,\s+CUDA\s+([\d.]+)\s*ms"
    )
    switch_iter_parser = re.compile(r"--- MyTimer: Switched to iteration (\d+) ---")

    for log_file in sorted(log_files):
        last_switch_iter = -1
        pending_runs = []
        current_run_stages = []

        with open(log_file, 'r') as f:
            for line in f:
                switch_match = switch_iter_parser.search(line)
                if switch_match:
                    new_switch_iter = int(switch_match.group(1))
                    if new_switch_iter == 1 and last_switch_iter > 1:
                        print(f"Info: Reset detected in {os.path.basename(log_file)}. Discarding data from iter {last_switch_iter} -> {new_switch_iter}.")
                        pending_runs = []
                    else:
                        for run in pending_runs:
                            all_runs_data[run['true_iter']][run['rank']].append(run['stages'])
                        pending_runs = []
                    last_switch_iter = new_switch_iter
                    continue

                stage_match = line_parser.match(line)
                if stage_match:
                    rank = int(stage_match.group(1))
                    stage_name = stage_match.group(2)
                    cuda_time_s = float(stage_match.group(3)) / 1000.0
                    
                    if stage_name.startswith('iteration-'):
                        true_iter_id = int(stage_name.split('-')[-1])
                        if current_run_stages:
                            pending_runs.append({'true_iter': true_iter_id, 'rank': rank, 'stages': current_run_stages})
                        current_run_stages = []
                    else:
                        current_run_stages.append({'name': stage_name, 'duration_s': cuda_time_s})
            
            for run in pending_runs:
                all_runs_data[run['true_iter']][run['rank']].append(run['stages'])
    return all_runs_data

def remove_outliers_iqr(data):
    """Removes outliers from a list of numbers using the IQR method."""
    # This function remains unchanged
    if len(data) < 4: return data, []
    data = np.array(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return clean_data.tolist(), outliers.tolist()

def analyze_and_average_runs(all_runs_data):
    """
    Analyzes runs, removes outliers, calculates averages, computes idle time,
    and now identifies and aggregates DiT stage times by their specific shape.
    """
    analysis_results = defaultdict(lambda: defaultdict(dict))
    averaged_data = defaultdict(dict)

    for true_iter, ranks_data in all_runs_data.items():
        # First pass: Calculate averaged data for each rank (this part is unchanged)
        for rank, runs in ranks_data.items():
            if not runs: continue
            if len(set(len(run) for run in runs)) > 1:
                print(f"Warning: Inconsistent stage count for iter {true_iter}, rank {rank}. Skipping.")
                continue

            num_stages = len(runs[0])
            averaged_stages_for_plot, stage_analysis_for_json = [], []

            for i in range(num_stages):
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
        
        # Second pass: Calculate makespan, idle times, and DiT times for the iteration
        if true_iter in averaged_data:
            rank_total_times = {rank: sum(s['duration_s'] for s in stages) for rank, stages in averaged_data[true_iter].items()}

            # ###################################################################################
            # ##### 新功能: 识别每个DiT形状并按形状统计耗时 #####
            # ###################################################################################
            for rank, stages in averaged_data[true_iter].items():
                # 使用 defaultdict 来方便地累加时间
                dit_times_by_shape = defaultdict(float)

                for stage in stages:
                    # 检查是否为DiT forward或backward阶段
                    if stage['name'].startswith('forward_single') or stage['name'].startswith('backward_bs'):
                        # 使用 format_stage_name 获取标准化的形状名称作为key
                        formatted_name = format_stage_name(stage['name'])
                        # 累加相同形状的DiT阶段的耗时
                        dit_times_by_shape[formatted_name] += stage['duration_s']
                
                # 将 defaultdict 转换为普通 dict 并对时间进行 round 操作，以便于JSON输出
                dit_times_by_shape_rounded = {name: round(time, 4) for name, time in dit_times_by_shape.items()}

                # 如果字典不为空，则将其存入最终的分析结果中
                if dit_times_by_shape_rounded:
                    rank_key = f"rank_{rank}"
                    analysis_results[f"iteration_{true_iter}"][rank_key]["dit_times_by_shape"] = dit_times_by_shape_rounded
            # ###################################################################################
            # ##### 新功能结束 #####
            # ###################################################################################

            if rank_total_times:
                iteration_makespan = max(rank_total_times.values())
                analysis_results[f"iteration_{true_iter}"]["iteration_total_time_s"] = round(iteration_makespan, 4)

                # --- Calculate and store active and idle time for each rank ---
                for rank, active_time in rank_total_times.items():
                    idle_time = iteration_makespan - active_time
                    rank_key = f"rank_{rank}"
                    analysis_results[f"iteration_{true_iter}"][rank_key]["active_time_s"] = round(active_time, 4)
                    analysis_results[f"iteration_{true_iter}"][rank_key]["idle_time_s"] = round(idle_time, 4)

    return analysis_results, averaged_data

def plot_iteration_gantt_chart(iteration_num, iteration_data, makespan, output_dir):
    """
    Generates and saves a Gantt chart, now including a visual representation of idle time.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ranks = sorted(iteration_data.keys())
    
    all_stage_names = {format_stage_name(stage['name']) for rank_data in iteration_data.values() for stage in rank_data}
    # Add "Idle Time" to the set for a consistent color
    all_stage_names.add("Idle Time")
    
    colors = cm.get_cmap('tab20', len(all_stage_names))
    color_map = {name: colors(i) for i, name in enumerate(all_stage_names)}
    # Use a specific, muted color for idle time
    color_map["Idle Time"] = '#d3d3d3' # Light grey

    for rank in ranks:
        current_time = 0.0
        for stage in iteration_data.get(rank, []):
            start_time, duration = current_time, stage['duration_s']
            formatted_name = format_stage_name(stage['name'])
            ax.barh(f"Rank {rank}", duration, left=start_time, height=0.6,
                    edgecolor='black', color=color_map[formatted_name], align='center')
            current_time += duration
        
        # --- NEW: Add the idle time bar ---
        idle_time = makespan - current_time
        if idle_time > 1e-4: # Add a small threshold to avoid tiny floating point bars
            ax.barh(f"Rank {rank}", idle_time, left=current_time, height=0.6,
                    edgecolor='grey', color=color_map["Idle Time"], hatch='..', align='center')

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title(f"Averaged Timeline of All Ranks for True Iteration {iteration_num}", fontsize=16, weight='bold')
    
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
    """Creates dummy log files for demonstration."""
    # This function remains unchanged
    print("Creating dummy log files for demonstration...")
    log0_content = """[2025-08-23 20:07:53] [Rank 0] [INFO] [next_iteration:260] - --- MyTimer: Switched to iteration 6 ---
[2025-08-23 20:28:33] [Rank 0] [INFO] [stop:254] - [Iter 1] Stage 'vae_forward_bs_1_f_129_h_1280_w_720_sp_6_distNone': CPU 5250.119ms, CUDA 5250.179ms
[2025-08-23 20:29:01] [Rank 0] [INFO] [stop:254] - [Iter 1] Stage 'iteration-6': CPU 33857.926ms, CUDA 33858.148ms
"""
    with open("timer_rank0.log", "w") as f: f.write(log_content)

if __name__ == "__main__":
    model, resolution = "cogvideox", "720p"
    base_log_dir = "dynamic_flex_exp_log"
    framework = "arachne"
    log_search_path = os.path.join(base_log_dir,framework ,model, resolution, "timing_rank*.log")
    output_figure_dir = os.path.join(base_log_dir, framework ,model, resolution,  "figures")
    
    all_runs_data = parse_and_group_runs(log_search_path)

    if all_runs_data is None:
        create_dummy_logs()
        all_runs_data = parse_and_group_runs("timing_rank*.log")
        output_figure_dir = "figures"

    if all_runs_data:
        analysis_results, averaged_data = analyze_and_average_runs(all_runs_data)
        
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
            # Pass the calculated makespan to the plotting function
            iter_key = f"iteration_{iter_num}"
            if iter_key in analysis_results and "iteration_total_time_s" in analysis_results[iter_key]:
                makespan = analysis_results[iter_key]["iteration_total_time_s"]
                plot_iteration_gantt_chart(iter_num, iter_data, makespan, output_figure_dir)
        print("\nAll charts generated successfully!")
    else:
        print("Could not process any log data.")