import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

# ==============================================================================
# ==============================================================================
THROUGHPUT_CONFIG = {
    'Wan-1.3B': {
        'base_model_name': 'wan1.3b',
        'resolution': '720p'
    },
    'CogVideoX-5B': {
        'base_model_name': 'cogvideox',
        'resolution': '720p',
    },
    'HunyuanVideo-13B':{
        'base_model_name': 'hunyuan-129',
        'resolution': '720p'
    },
    'hunyuan-53':{
        'base_model_name': 'hunyuan-53', 'resolution': '720p'
    },
    'hunyuan-81':{
        'base_model_name': 'hunyuan-81', 'resolution': '720p'
    },
    'hunyuan-129':{
        'base_model_name': 'hunyuan-129',
        'resolution': '720p'
    },
    'hunyuan-105-frames': {
        'base_model_name': 'hunyuan-129',
        'resolution': '720p'
    },
    'hunyuan-4nodes':{
        'base_model_name': 'hunyuan-4nodes',
        'resolution': '720p'
    },
    'hunyuan-8nodes':{
        'base_model_name': 'hunyuan-8nodes',
        'resolution': '720p'
    },
}

MANUAL_OVERRIDES = {
    # 'CogVideoX-5B': { 'Megatron-LM': 1.21, 'DeepSpeed': 1.23, 'Arachne': 1.6335 },
    # 'HunyuanVideo-13B': { 'Megatron-LM': 0.35, 'DeepSpeed': 0.3561, 'Arachne': 0.4921 },
    'hunyuan-129': { 'Megatron-LM': 0.35, 'DeepSpeed': 0.3561, 'Arachne': 0.4921 },
    'hunyuan-105-frames': { 'Megatron-LM': 0.39, 'DeepSpeed': 0.40, 'Arachne': 0.5421 },
    'hunyuan-4nodes': { 'DeepSpeed': 0.47, },
    'hunyuan-8nodes': { 'Megatron-LM': 0.76, 'DeepSpeed': 0.75, 'Arachne': 1.0296 },
}

SYSTEMS_TO_COMPARE = ['Megatron-LM', 'DeepSpeed', 'Arachne']
AESTHETICS = { 'color_map': {'Megatron-LM': '#ACD7E5', 'DeepSpeed': '#EFD2BA', 'Arachne': '#FE8B01',}, 'hatch_map': {'Megatron-LM': '///', 'DeepSpeed': '...', 'Arachne': 'xxx'}}

# ==============================================================================
# ==============================================================================
def create_dummy_files():
    pass

# ==============================================================================
# ==============================================================================
def load_iteration_times(framework, base_model, resolution):
    framework_folder = 'arachne' if framework == 'Arachne' else framework.lower()
    base_dir = 'dynamic_flex_exp_log' if framework == 'Arachne' else 'baseline_log'
    filepath = os.path.join(base_dir, framework_folder, base_model, resolution, 'iteration_times_summary.json' if framework_folder == "deepspeed" else 'figures/iteration_times_summary.json')
    try:
        with open(filepath, 'r') as f: return {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print(f"info: times file not found, skipping: {filepath}"); return None

def load_total_frames(framework, base_model, resolution):
    filepath = os.path.join("simulation_data", f"total_frames_{base_model}_{resolution}.txt")
    frames_data = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = re.search(r'iteration\s+(\d+)\s+:\s+(\d+)', line)
                if match: frames_data[int(match.groups()[0])] = int(match.groups()[1])
            return frames_data
    except FileNotFoundError:
        print(f"info: frames file not found, skipping: {filepath}"); return None

# ==============================================================================
# ==============================================================================
def plot_single_subplot(ax, throughput_data, annotation_data, title, tick_map=None):
    original_labels = list(throughput_data.keys())
    display_labels = [tick_map.get(k, k) for k in original_labels] if tick_map else original_labels
    systems = SYSTEMS_TO_COMPARE
    data_for_plot = {sys: [throughput_data[case].get(sys, 0) for case in original_labels] for sys in systems}
    x = np.arange(len(original_labels))
    width = 0.25
    n_systems = len(systems)

    for i, system in enumerate(systems):
        offset = width * (i - (n_systems - 1) / 2)
        rects = ax.bar(x + offset, data_for_plot[system], width,
                       label=system, color=AESTHETICS['color_map'].get(system),
                       hatch=AESTHETICS['hatch_map'].get(system), edgecolor='black', zorder=3)

        for j, rect in enumerate(rects):
            height = rect.get_height()
            case_name = original_labels[j]
            label_text = f'{height:.2f}'
            if system == 'Arachne' and case_name in annotation_data:
                ann = annotation_data[case_name]
                vs_m = ann.get('vs_megatron', 0)
                vs_d = ann.get('vs_deepspeed', 0)
                if vs_m > 0 and vs_d > 0:
                    label_text += f'\n({vs_m:.2f}x, {vs_d:.2f}x)'
            
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5),
                        textcoords="offset points", ha='center', va='bottom',
                        fontsize=12, fontweight='bold', multialignment='center', linespacing=1.2)

    ax.set_title(title, fontsize=18, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    max_val = max(val for sys_data in data_for_plot.values() for val in sys_data) if any(data_for_plot.values()) else 1
    ax.set_ylim(0, max_val * 1.4)

def plot_throughput_subplots(data1, annotations1, map1, data2, annotations2, map2, data3, annotations3, map3):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'hatch.linewidth': 0.5})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_single_subplot(axes[0], data1, annotations1, "Across Different Models", tick_map=map1)
    axes[0].set_ylabel('Throughput (Frames/GPU/Second)', fontsize=16, fontweight='bold', labelpad=10)
    plot_single_subplot(axes[1], data2, annotations2, "Across Different Max Frames", tick_map=map2)
    plot_single_subplot(axes[2], data3, annotations3, "Across Different Cluster Sizes", tick_map=map3)

    handles = [Patch(facecolor=AESTHETICS['color_map'][s], hatch=AESTHETICS['hatch_map'][s], edgecolor='black') for s in SYSTEMS_TO_COMPARE]
    
    fig.legend(handles, SYSTEMS_TO_COMPARE, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(SYSTEMS_TO_COMPARE), fontsize=16, frameon=False, prop={'weight': 'bold'})

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_filename = 'throughput_comparison_subplots.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n[SUCCESS] throughput comparison saved to {output_filename}")


# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    final_throughput_data = {}
    print("="*60 + f"\nStep 1: compute per-model absolute throughput...\n" + "="*60)

    for case_name, config in THROUGHPUT_CONFIG.items():
        print(f"\n--- processing case: {case_name} ---")

        if case_name == 'hunyuan-4nodes':
            gpu_count_for_case = 32
        elif case_name == 'hunyuan-8nodes':
            gpu_count_for_case = 64
        else:
            gpu_count_for_case = 16
        
        print(f"   -> GPU count: {gpu_count_for_case}")

        case_throughputs = {}
        for system in SYSTEMS_TO_COMPARE:
            override_value = MANUAL_OVERRIDES.get(case_name, {}).get(system)
            avg_throughput = None
            if override_value is not None:
                print(f"   -> {system}: manual override: {override_value:.2f} frames/GPU/s")
                avg_throughput = override_value
            else:
                times_data = load_iteration_times(system, config['base_model_name'], config['resolution'])
                frames_data = load_total_frames(system, config['base_model_name'], config['resolution'])
                if times_data and frames_data:
                    common_iters = sorted(list(set(times_data.keys()) & set(frames_data.keys())))
                    if common_iters:
                        total_frames = sum(frames_data[it] for it in common_iters)
                        total_time = sum(times_data[it] for it in common_iters if times_data[it] > 0)
                        if total_time > 0:
                            avg_throughput = total_frames / (gpu_count_for_case * total_time)
                print(f"   -> {system}: total frames={total_frames}, total time={total_time:.2f}, throughput={avg_throughput:.4f} frames/GPU/s")

            # else:
            #     times_data = load_iteration_times(system, config['base_model_name'], config['resolution'])
            #     frames_data = load_total_frames(system, config['base_model_name'], config['resolution'])
            #     if times_data and frames_data:
            #         common_iters = sorted(list(set(times_data.keys()) & set(frames_data.keys())))
            #         if common_iters:
            #             throughputs = [frames_data[it] / (gpu_count_for_case * times_data[it]) for it in common_iters if times_data[it] > 0]
            #             if throughputs:
            #                 avg_throughput = np.mean(throughputs)

            if avg_throughput is not None:
                case_throughputs[system] = avg_throughput
        if case_throughputs:
            final_throughput_data[case_name] = case_throughputs

    if final_throughput_data:
        print("\n" + "="*60 + "\nStep 2: compute Arachne speedups for annotations...\n" + "="*60)
        annotation_data = {}
        for case_name, system_throughputs in final_throughput_data.items():
            arachne_tp = system_throughputs.get('Arachne')
            megatron_tp = system_throughputs.get('Megatron-LM')
            deepspeed_tp = system_throughputs.get('DeepSpeed')
            case_annotations = {}
            if arachne_tp and megatron_tp and megatron_tp > 0: case_annotations['vs_megatron'] = arachne_tp / megatron_tp
            if arachne_tp and deepspeed_tp and deepspeed_tp > 0: case_annotations['vs_deepspeed'] = arachne_tp / deepspeed_tp
            if case_annotations: annotation_data[case_name] = case_annotations

        keys_subplot1 = ['Wan-1.3B', 'CogVideoX-5B', 'HunyuanVideo-13B']
        keys_subplot2 = ['hunyuan-53', 'hunyuan-81', 'hunyuan-105-frames']
        keys_subplot3 = ['hunyuan-129', 'hunyuan-4nodes', 'hunyuan-8nodes']
        data_subplot1 = {k: final_throughput_data[k] for k in keys_subplot1 if k in final_throughput_data}
        annotations1 = {k: annotation_data[k] for k in keys_subplot1 if k in annotation_data}
        data_subplot2 = {k: final_throughput_data[k] for k in keys_subplot2 if k in final_throughput_data}
        annotations2 = {k: annotation_data[k] for k in keys_subplot2 if k in annotation_data}
        data_subplot3 = {k: final_throughput_data[k] for k in keys_subplot3 if k in final_throughput_data}
        annotations3 = {k: annotation_data[k] for k in keys_subplot3 if k in annotation_data}
        xtick_map_subplot2 = {'hunyuan-53': '53 frames', 'hunyuan-81': '81 frames', 'hunyuan-105-frames': '105 frames'}
        xtick_map_subplot3 = {'hunyuan-129': '2 nodes', 'hunyuan-4nodes': '4 nodes', 'hunyuan-8nodes': '8 nodes'}

        print("\n" + "="*60 + "\nStep 3: render the figure...\n")
        plot_throughput_subplots(
            data_subplot1, annotations1, None,
            data_subplot2, annotations2, xtick_map_subplot2,
            data_subplot3, annotations3, xtick_map_subplot3
        )
    else:
        print("\nno data computed; cannot render the figure.")
