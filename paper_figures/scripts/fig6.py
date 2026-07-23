"""Average iteration time / speedup comparison (paper Fig. 6).

- MASTER_CONFIG defines the experiment cases (base_model_name, resolution,
  optional max_frames).
- Reads each system's iteration_times_summary.json:
    Arachne:     dynamic_flex_exp_log/arachne/<model>/<res>/[<frames>/]figures/
    Megatron-LM: baseline_log/megatron-lm/<model>/<res>/[<frames>/]figures/
    FlexSP:      baseline_log/flex_sp/... (optional)
    DeepSpeed:   baseline_log/deepspeed/... (optional; NO figures/ level)
- Averages over the iterations COMMON to all systems; with
  USE_AVERAGE_OF_ITERATION_SPEEDUPS the annotation is the mean per-iteration
  speedup; MANUAL_OVERRIDES can pin annotation values.
- Renders the bar charts per SCENARIOS_TO_PLOT and regenerates the
  plot_data_*.json files used by the paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.patches import Patch
from paper_plot_style import apply_paper_style

# =============================================================================
# =============================================================================

USE_AVERAGE_OF_ITERATION_SPEEDUPS = True


MANUAL_OVERRIDES = {
}

AESTHETICS = {
    'system_order': ['Megatron-LM', 'DeepSpeed', 'FlexSP' ,'Arachne'],
 'color_map': {
        'Megatron-LM': '#7f8c8d',
        'DeepSpeed':   '#4e79a7',
        'FlexSP':      '#59a14f',
            'Arachne': '#e0802c'

    },
'hatch_map': {
        'Megatron-LM': '/',
        'DeepSpeed': '..',
        'FlexSP': '||',
        'Arachne': None,
    },
    'primary_baseline': 'Megatron-LM'
}

STAGE2_FALLBACK_DATA = {
    'Arachne_times': [29.03, 36.44, 22.79, 41.76, 28.84, 40.5, 19.83, 23.45, 18.80, 29.90, 39.70, 26.80, 29.43, 24.36, 49.13, 44.77],
    'baselines': {
        'Megatron-LM': [35.60, 59.20, 24.20, 59.36, 58.67, 58.67, 18.743, 31.79, 16.35, 59.47, 62.29, 39.84, 60.26, 31.25, 55.15, 49.02],
        'DeepSpeed': [38.61, 56.22, 27.55, 55.67, 56.19, 52.17, 22.50, 33.40, 19.08, 56.06, 55.57, 40.75, 56.2, 33.39, 52.40, 45.49]
    }
}

MASTER_CONFIG = {
    'Stage1': {
        'subplot_title': 'Stage 1 (Dataset=WebVid, Resolution=360p)',
        'loader_config': {'base_model_name': 'hunyuan-129', 'resolution': '360p'},
    },
    'Stage2': {
        'subplot_title': 'Stage 2 (Dataset=Koala, Resolution=720p)',
        'loader_config': {'base_model_name': 'hunyuan-129', 'resolution': '720p'},
        'fallback_data': STAGE2_FALLBACK_DATA
    },
    'Stage3': {
        'subplot_title': 'Stage 3 (Dataset=Lynx, Resolution=1080p)',
        'loader_config': {'base_model_name': 'hunyuan-129', 'resolution': '1080p'},
        'fallback_data': {
            'Arachne_times': [41.195, 27.19, 35.45, 40.42, 35.12, 21.34, 45.0, 42.0, 48.0, 41.0, 29.0, 44.0, 39.0],
            'baselines': { 'Megatron-LM': [62.71, 35.97, 57.05, 62.58, 60.03, 25.1, 68.4, 60.13, 68.54, 60.04, 39.59, 62.06, 46.73], 'DeepSpeed': [58.1, 33.2, 54.5, 59.3, 57.2, 24.0, 65.1, 58.2, 66.3, 57.9, 38.1, 59.5, 45.2] }
        }
    },
    'wan-1.3b': {
        'subplot_title': 'Wan-1.3B',
        'loader_config': {'base_model_name': 'wan1.3b', 'resolution': '720p'},
        'fallback_data': {}
    },
    'cogvideox-5b': {
        'subplot_title': 'CogVideoX-5B',
        'loader_config': {'base_model_name': 'cogvideox', 'resolution': '720p'},
        'fallback_data': {}
    },
    'hunyuanvideo-13b': {
        'subplot_title': 'Hunyuan-13B',
        'loader_config': {'base_model_name': 'hunyuan-129', 'resolution': '720p'},
        'fallback_data': STAGE2_FALLBACK_DATA
    },
    'Hunyuan13B_720p_53frames': {
        'subplot_title': 'Frames=53',
        'loader_config': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': 53},
    },
    'Hunyuan13B_720p_81frames': {
        'subplot_title': 'Frames=81',
        'loader_config': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': 81},
        'fallback_data': STAGE2_FALLBACK_DATA
    },
    'Hunyuan13B_720p_129frames': {
        'subplot_title': 'Frames=129',
        'loader_config': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': 129},
        'fallback_data': STAGE2_FALLBACK_DATA
    }
}

# =============================================================================
# =============================================================================

def load_data_from_json(loader_config):
    base_model = loader_config.get('base_model_name')
    resolution = loader_config.get('resolution')

    max_frames = loader_config.get('max_frames', None)

    def join_with_optional_frames(*parts):
        if max_frames is None:
            return os.path.join(*parts)
        return os.path.join(*parts, str(max_frames))
    # ======================================================================

    paths = {
        # Arachne
        'Arachne': os.path.join(
            "dynamic_flex_exp_log", "arachne",
            *join_with_optional_frames(base_model, resolution).split(os.sep),
            "figures", "iteration_times_summary.json"
        ),

        # Megatron-LM
        'Megatron-LM': os.path.join(
            "baseline_log", "megatron-lm",
            *join_with_optional_frames(base_model, resolution).split(os.sep),
            "figures", "iteration_times_summary.json"
        ),

        # FlexSP (optional)
        'FlexSP': os.path.join(
            "baseline_log", "flex_sp",
            *join_with_optional_frames(base_model, resolution).split(os.sep),
            "figures", "iteration_times_summary.json"
        ),

        # DeepSpeed (optional)
        'DeepSpeed': os.path.join(
            "baseline_log", "deepspeed",
            *join_with_optional_frames(base_model, resolution).split(os.sep),
            "iteration_times_summary.json"
        )
    }

    loaded_data = {}

    # new add FlexSP
    flexsp_path = paths['FlexSP']
    print(f"        -> Attempting to load (optional): {flexsp_path}")
    try:
        with open(flexsp_path, 'r', encoding="utf-8-sig") as f:
            loaded_data['FlexSP'] = json.load(f)
    except FileNotFoundError:
        print(f"        -> SKIPPED: FlexSP file not found.")
        pass

    for system in ['Arachne', 'Megatron-LM']:
        filepath = paths[system]
        print(f"        -> Attempting to load: {filepath}")
        try:
            with open(filepath, 'r', encoding="utf-8-sig") as f:
                loaded_data[system] = json.load(f)
        except FileNotFoundError:
            print(f"        -> FAILED: Required file not found.")
            return None

    deepspeed_path = paths['DeepSpeed']
    print(f"        -> Attempting to load (optional): {deepspeed_path}")
    try:
        with open(deepspeed_path, 'r', encoding="utf-8-sig") as f:
            loaded_data['DeepSpeed'] = json.load(f)
    except FileNotFoundError:
        print(f"        -> SKIPPED: Optional file not found.")
        pass

    key_sets = [set(int(k) for k in data.keys()) for data in loaded_data.values()]
    common_iterations = sorted(list(set.intersection(*key_sets)))
    if not common_iterations:
        return None

    new_data = {'Arachne_times': [loaded_data['Arachne'][str(i)] for i in common_iterations], 'baselines': {}}
    if 'Megatron-LM' in loaded_data:
        new_data['baselines']['Megatron-LM'] = [loaded_data['Megatron-LM'][str(i)] for i in common_iterations]
    if 'DeepSpeed' in loaded_data:
        new_data['baselines']['DeepSpeed'] = [loaded_data['DeepSpeed'][str(i)] for i in common_iterations]
    if 'FlexSP' in loaded_data:
        new_data['baselines']['FlexSP'] = [loaded_data['FlexSP'][str(i)] for i in common_iterations]
    return new_data

def process_single_experiment(exp_key, exp_config):
    print(f"\n--- Processing: {exp_key} ---")
    data_to_use = load_data_from_json(exp_config.get('loader_config'))
    if data_to_use is None:
        print(" -> JSON load failed. Using fallback data.")
        data_to_use = exp_config.get('fallback_data')
        if not data_to_use or not data_to_use.get('Arachne_times'):
            print(" -> No data available. Skipping.")
            return None
    else:
        print(" -> Successfully loaded from JSON.")
    
    bar_heights_actual_times = {}
    arachne_annotations = {}

    if USE_AVERAGE_OF_ITERATION_SPEEDUPS:
        print(" -> Calculating: Average of Per-Iteration Times and Speedups.")
        arachne_times_np = np.array(data_to_use['Arachne_times'])
        
        if 'Megatron-LM' not in data_to_use['baselines']:
            print(" -> ERROR: Primary baseline 'Megatron-LM' not found.")
            return None
        megatron_times_np = np.array(data_to_use['baselines']['Megatron-LM'])

        bar_heights_actual_times['Arachne'] = np.mean(arachne_times_np)
        bar_heights_actual_times['Megatron-LM'] = np.mean(megatron_times_np)

        vs_m = np.mean(np.divide(megatron_times_np, arachne_times_np,
                                  out=np.ones_like(arachne_times_np, dtype=float),
                                  where=arachne_times_np!=0))
        arachne_annotations['vs_megatron'] = vs_m

        if 'DeepSpeed' in data_to_use['baselines']:
            deepspeed_times_np = np.array(data_to_use['baselines']['DeepSpeed'])
            bar_heights_actual_times['DeepSpeed'] = np.mean(deepspeed_times_np)
            
            vs_d = np.mean(np.divide(deepspeed_times_np, arachne_times_np,
                                      out=np.ones_like(arachne_times_np, dtype=float),
                                      where=arachne_times_np!=0))
            arachne_annotations['vs_deepspeed'] = vs_d

        if 'FlexSP' in data_to_use['baselines']:
            flex_times_np = np.array(data_to_use['baselines']['FlexSP'])
            bar_heights_actual_times['FlexSP'] = np.mean(flex_times_np)
            
            vs_f = np.mean(np.divide(flex_times_np, arachne_times_np,
                                    out=np.ones_like(arachne_times_np, dtype=float),
                                    where=arachne_times_np!=0))
            arachne_annotations['vs_flexsp'] = vs_f
    else:
        print("ERROR: This script is designed to work with USE_AVERAGE_OF_ITERATION_SPEEDUPS = True.")
        return None
            
    return {
        'subplot_title': exp_config['subplot_title'],
        'bar_heights_actual_times': bar_heights_actual_times,
        'arachne_annotations': arachne_annotations
    }

def plot_single_chart(ax, data, aesthetics):
    labels = aesthetics['system_order']
    values = [data['bar_heights_actual_times'].get(s, 0) for s in labels]
    colors = [aesthetics['color_map'].get(s, 'grey') for s in labels]
    hatches = [aesthetics['hatch_map'].get(s, None) for s in labels]
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', hatch=hatches, width=0.7,zorder=3)
    
    # =======================================================================
    # =======================================================================
    for bar, label in zip(bars, labels):
        if label == 'Arachne':
            bar.set_linewidth(2.2)
            bar.set_edgecolor('black')
            break
    # =======================================================================

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='#555555')

            if label == 'Arachne':
                annotations = data['arachne_annotations']
                
                speedup_texts = []
                
                if 'vs_megatron' in annotations:
                    speedup_texts.append(rf"$\mathbf{{{annotations['vs_megatron']:.2f}\times}}$")
                
                if 'vs_deepspeed' in annotations:
                    speedup_texts.append(rf"$\mathbf{{{annotations['vs_deepspeed']:.2f}\times}}$")
                
                if 'vs_flexsp' in annotations:
                    speedup_texts.append(rf"$\mathbf{{{annotations['vs_flexsp']:.2f}\times}}$")
                
                final_speedup_text = ", ".join(speedup_texts)

                if final_speedup_text:
                    # ax.annotate(final_speedup_text,
                    #             xy=(bar.get_x() + bar.get_width() / 2, height),
                    #             xytext=(0, 30), textcoords="offset points",
                    #             color='red',
                    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
                    ax.annotate(
                        final_speedup_text,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(38, 35),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9.5,
                        fontweight='bold',
                        rotation=8,
                        color='red',
                        clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="white",
                            ec="none",
                            alpha=0.9
                        )
                    )


    ax.set_title(data['subplot_title'], fontsize=14, pad=15, fontweight='bold')
    
    ax.tick_params(axis='x', labelsize=10)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

from matplotlib.legend_handler import HandlerPatch

def create_comparison_figure(figure_data, aesthetics, figure_title, filename):
    n_subplots = len(figure_data)
    if n_subplots == 0: return

    # old standalone style        
    # plt.style.use('seaborn-v0_8-paper')
    # plt.rcParams.update({
    #     'font.family': 'sans-serif',
    #     'mathtext.fontset': 'dejavusans',

    #     'pdf.fonttype': 42,
    #     'ps.fonttype': 42,
    # })

    fig, axes = plt.subplots(1, n_subplots, figsize=(5.5 * n_subplots, 4.0), sharey=False)
    if n_subplots == 1: axes = [axes]
    apply_paper_style(figsize=(5.5 * n_subplots, 4.0))
    


    
    for ax, data in zip(axes, figure_data):
        plot_single_chart(ax, data, aesthetics)
        
        if data and 'bar_heights_actual_times' in data and data['bar_heights_actual_times']:
            local_max_time = max(data['bar_heights_actual_times'].values())
            ax.set_ylim(bottom=0, top=local_max_time * 1.30)
        else:
            ax.set_ylim(bottom=0)
    
    y_label = "Average Iteration Time (s)"
    axes[0].set_ylabel(y_label, fontsize=15, fontweight='bold')
    
    # handles = []
    # for system_name in aesthetics['system_order']:
    #     is_arachne = (system_name == 'Arachne')
    #     line_width = 2.2 if is_arachne else 1.0
    #     patch = Patch(
    #         facecolor=aesthetics['color_map'][system_name], 
    #         hatch=aesthetics['hatch_map'][system_name], 
    #         edgecolor='black', 
    #         label=system_name,
    #         linewidth=line_width
    #     )
    #     handles.append(patch)

    # # =======================================================================
    # # =======================================================================
    # class custom_handler(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                        xdescent, ydescent, width, height, fontsize,
    #                        trans):
    #         patch = super().create_artists(legend, orig_handle,
    #                                         xdescent, ydescent, width, height, fontsize, trans)[0]
    #         patch.set_hatch(orig_handle.get_hatch())
    #         return [patch]

    # fig.legend(handles=handles, 
    #            loc='upper center', 
    #            bbox_to_anchor=(0.5, 1.02), 
    #            ncol=len(aesthetics['system_order']), 
    #            fontsize=14, 
    #            frameon=False,
    #            handler_map={Patch: custom_handler()})
    # # =======================================================================

    plt.tight_layout(rect=[0, 0, 1, 0.935])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n[PAPER FIGURE] Chart '{figure_title}' has been saved as '{filename}'")


# =============================================================================
# =============================================================================
if __name__ == "__main__":
    SCENARIOS_TO_PLOT = [
        {
            'figure_title': 'End-to-End Performance on Three-Stage Progressive Training',
            'experiment_keys': ['Stage1', 'Stage2', 'Stage3'],
            'output_filename_pdf': 'avg_iteration_time_across_stage_resolutions.pdf',
            'output_data_json': 'plot_data_stages_avg_time.json'
        },
        {
            'figure_title': 'Performance Across Model Sizes (at 720p)',
            'experiment_keys': ['wan-1.3b', 'cogvideox-5b', 'hunyuanvideo-13b'],
            'output_filename_pdf': 'avg_iteration_time_across_model_sizes.pdf',
            'output_data_json': 'plot_data_models_avg_time.json'
        },
        {
            'figure_title': 'Performance Across Max Frame Windows (Hunyuan-13B at 720p)',
            'experiment_keys': ['Hunyuan13B_720p_53frames', 'Hunyuan13B_720p_81frames', 'Hunyuan13B_720p_129frames'],
            'output_filename_pdf': 'avg_iteration_time_across_frame_windows.pdf',
            'output_data_json': 'plot_data_frames_avg_time.json'
        }
    ]

    for scenario in SCENARIOS_TO_PLOT:
        print(f"\n{'='*30}\nGenerating assets for: {scenario['figure_title']}\n{'='*30}")
        
        processed_data_for_figure = []
        for exp_key in scenario['experiment_keys']:
            if exp_key in MASTER_CONFIG:
                print(f"doing exp_key = {exp_key}")
                result = process_single_experiment(exp_key, MASTER_CONFIG[exp_key])
                if result:
                    if exp_key in MANUAL_OVERRIDES:
                        print(f" -> Applying manual override for '{exp_key}'...")
                        override_values = MANUAL_OVERRIDES[exp_key]
                        for key, value in override_values.items():
                            if 'arachne_annotations' in result and key in result['arachne_annotations']:
                                result['arachne_annotations'][key] = value
                                print(f"       -> Overrode annotation '{key}' to {value}")
                    processed_data_for_figure.append(result)
            else:
                print(f"Warning: Experiment key '{exp_key}' not found in MASTER_CONFIG.")
        
        create_comparison_figure(
            figure_data=processed_data_for_figure,
            aesthetics=AESTHETICS,
            figure_title=scenario['figure_title'],
            filename=scenario['output_filename_pdf']
        )
        
        data_to_save = {
            "figure_title": scenario['figure_title'],
            "aesthetics": AESTHETICS,
            "figure_data": processed_data_for_figure
        }
        with open(scenario['output_data_json'], 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\n[INTERMEDIATE DATA] Plot data saved to '{scenario['output_data_json']}'")

