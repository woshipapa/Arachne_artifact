
"""
   - Arachne: dynamic_flex_exp_log/arachne/<base_model>/<resolution>/[<max_frames>/]figures/gpu_utilization_summary.json
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==============================================================================
# ==============================================================================

ANALYSIS_CONFIG = {
    'Stage 1': {
        'base_model_name': 'hunyuan-129',
        'resolution': '360p'
    },
    'Stage 2': {
        'base_model_name': 'hunyuan-129',
        'resolution': '720p',
    },
    'Stage 3':{
        'base_model_name': 'hunyuan-129',
        'resolution': '1080p'
    },

    # 'Frames=53': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': 53},
    # 'Frames=81': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': 81},
}

MANUAL_OVERRIDES = {
    'Stage 3': {'DeepSpeed': 0.5418}
}

SYSTEMS_TO_COMPARE = ['Megatron-LM', 'DeepSpeed', 'FlexSP', 'Arachne']  # === NEW ===


# ==============================================================================
# ==============================================================================

def create_dummy_utilization_files():
    print("--- Creating dummy utilization files for demonstration ---")
    for case, config in ANALYSIS_CONFIG.items():
        for system in SYSTEMS_TO_COMPARE:
            framework_folder = 'arachne' if system == 'Arachne' else system.lower()
            # base_dir = 'dynamic_flex_exp_log' if system == 'Arachne' else 'baseline_log'
            base_dir = "experiment_data"

            if system == 'Arachne':
                dummy_util = {"iteration_0": 0.85, "iteration_1": 0.88, "iteration_2": 0.86}
            elif system == 'Megatron-LM':
                dummy_util = {"iteration_0": 0.65, "iteration_1": 0.68, "iteration_2": 0.64}
            elif system == 'FlexSP':  # === NEW ===
                dummy_util = {"iteration_0": 0.70, "iteration_1": 0.73, "iteration_2": 0.71}
            else:  # DeepSpeed
                dummy_util = {"iteration_0": 0.75, "iteration_1": 0.72, "iteration_2": 0.74}

            base_model = config['base_model_name']
            resolution = config['resolution']
            max_frames = config.get('max_frames', None)

            if max_frames is None:
                path = os.path.join(base_dir, framework_folder, base_model, resolution,
                                    'figures', 'gpu_utilization_summary.json')
            else:
                path = os.path.join(base_dir, framework_folder, base_model, resolution, str(max_frames),
                                    'figures', 'gpu_utilization_summary.json')
            # ====================================

            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump(dummy_util, f, indent=4)
                print(f"Created dummy file: {path}")

# ==============================================================================
# ==============================================================================

def _get_framework_folder(framework: str) -> str:
    if framework == 'Arachne':
        return 'arachne'
    if framework == 'Megatron-LM':
        return 'megatron-lm'
    if framework == 'FlexSP':
        return 'flex_sp'
    if framework == 'DeepSpeed':
        return 'deepspeed'
    return framework.lower()

def load_and_average_utilization(framework, base_model, resolution, max_frames=None):
    framework_folder = _get_framework_folder(framework)

    if framework == 'Arachne':
        base_dir = os.path.join('dynamic_flex_exp_log', 'arachne')
    else:
        base_dir = os.path.join('baseline_log', framework_folder)
    # =======================

    if max_frames is None:
        filepath = os.path.join(
            base_dir,
            base_model,
            resolution,
            'figures',
            'gpu_utilization_summary.json'
        )
    else:
        filepath = os.path.join(
            base_dir,
            base_model,
            resolution,
            str(max_frames),
            'figures',
            'gpu_utilization_summary.json'
        )

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        utilization_values = list(data.values())

        if not utilization_values:
            print(f"warn: '{filepath}' is empty; cannot average.")
            return None

        average_utilization = np.mean(utilization_values)
        print(f"loaded {filepath} -> avg utilization: {average_utilization:.2%}")
        return average_utilization

    except FileNotFoundError:
        print(f"info: file not found, skipping: {filepath}")
        return None
    except Exception as e:
        print(f"error reading {filepath}: {e}")
        return None

# ==============================================================================
# ==============================================================================

def plot_utilization_comparison(plot_data):
    labels = list(plot_data.keys())
    systems = SYSTEMS_TO_COMPARE
    
    data_for_plot = {sys: [] for sys in systems}
    for case_name in labels:
        for sys in systems:
            util = plot_data[case_name].get(sys, 0)
            idle = max(0.0, 1.0 - util)
            data_for_plot[sys].append(idle * 100)


    x = np.arange(len(labels))
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Verdana'],
        'mathtext.fontset': 'dejavusans',
        'pdf.fonttype': 42
    })
    plt.rcParams['hatch.linewidth'] = 0.8
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.2
    n_systems = len(systems)

    colors = {
        'Megatron-LM': '#ACD7E5',
        'DeepSpeed':   '#EFD2BA',
        'FlexSP':      '#59a14f',  # === NEW ===
        'Arachne':     '#FE8B01',
    }
    hatches = {
        'Megatron-LM': '/',
        'DeepSpeed': '..',
        'FlexSP': '||',  # === NEW ===
        'Arachne': None
    }
    
    all_rects = {}
    for i, system in enumerate(systems):
        offset = width * (i - (n_systems - 1) / 2)

        bar_linewidth = 2.2 if system == 'Arachne' else 0.7
        rects = ax.bar(x + offset, data_for_plot[system], width,
                       label=system, color=colors.get(system),
                       hatch=hatches.get(system), edgecolor='black', linewidth=bar_linewidth,
                       zorder=3)
        all_rects[system] = rects

    for i in range(len(labels)):
        last_height = -1
        vertical_offset_increment = 6
        base_padding = 3

        for system_idx, system in enumerate(systems):
            rect = all_rects[system][i]
            height = rect.get_height()
            
            current_padding = base_padding
            if system_idx > 0 and abs(height - last_height) < 2:
                current_padding += vertical_offset_increment

            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, current_padding),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color="#555555")
            
            last_height = height

    # ax.set_ylabel('Average GPU Utilization (%)', fontsize=16, fontweight='bold', labelpad=5)
    ax.set_ylabel('Average GPU Idle Ratio (%)', fontsize=16, fontweight='bold', labelpad=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
              ncol=len(systems), fontsize=14, frameon=False,
              columnspacing=1.5, handletextpad=0.5)

    # ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    max_val = 0
    for sys_data in data_for_plot.values():
        if len(sys_data) > 0:
            max_val = max(max_val, max(sys_data))
    ax.set_ylim(0, max_val * 1.12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # output_filename = 'average_gpu_utilization_comparison.pdf'
    output_filename = 'average_gpu_idle_ratio_comparison.pdf'

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    
    print(f"\n[SUCCESS] comparison figure saved to {output_filename}")


# ==============================================================================
# ==============================================================================

if __name__ == "__main__":

    # create_dummy_utilization_files()

    results_for_plot = {}

    print("="*60)
    print("computing average GPU utilization per case...")
    print("="*60)

    for case_name, config in ANALYSIS_CONFIG.items():
        print(f"\n--- processing case: {case_name} ---")

        case_results = {}
        for system in SYSTEMS_TO_COMPARE:

            override_value = MANUAL_OVERRIDES.get(case_name, {}).get(system)

            if override_value is not None:
                print(f"info: manual override for [{case_name}][{system}]: {override_value:.2%}")
                avg_util = override_value
            else:
                avg_util = load_and_average_utilization(
                    system,
                    config['base_model_name'],
                    config['resolution'],
                    max_frames=config.get('max_frames', None)
                )

            if avg_util is not None:
                case_results[system] = avg_util

        if case_results:
            results_for_plot[case_name] = case_results

    if results_for_plot:
        print("\n" + "="*60)
        print("done; final average utilization (overrides applied):")
        print(json.dumps(results_for_plot, indent=4))
        print("="*60)

        plot_utilization_comparison(results_for_plot)
    else:
        print("\nno data loaded; cannot render the figure.")
        print("check ANALYSIS_CONFIG and that the JSON files exist.")
