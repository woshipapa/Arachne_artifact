"""
   - Arachne: dynamic_flex_exp_log/arachne/<base_model>/<resolution>/[<max_frames>/]figures/iteration_times_summary.json
   - Megatron-LM: baseline_log/megatron-lm/<base_model>/<resolution>/[<max_frames>/]figures/iteration_times_summary.json
"""


from paper_plot_style import apply_paper_style
apply_paper_style(figsize=(18, 4.5))
# <<< TEMPLATE

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch

# ==============================================================================
# ==============================================================================

THROUGHPUT_CONFIG = {
    'Wan-1.3B': {'base_model_name': 'wan1.3b', 'resolution': '720p'},
    'CogVideoX-5B': {'base_model_name': 'cogvideox', 'resolution': '720p'},
    'HunyuanVideo-13B': {'base_model_name': 'hunyuan-129', 'resolution': '720p'},
    'hunyuan-53': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': '53'},
    'hunyuan-81': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': '81'},
    'hunyuan-105-frames': {'base_model_name': 'hunyuan', 'resolution': '720p', 'max_frames': '105'},
    'hunyuan-2nodes': {'base_model_name': 'hunyuan-129', 'resolution': '720p'},
    'hunyuan-4nodes': {'base_model_name': 'hunyuan-4nodes', 'resolution': '720p'},
    'hunyuan-8nodes': {'base_model_name': 'hunyuan-8nodes', 'resolution': '720p'},
}

SYSTEMS_ORDER = ['Megatron-LM', 'DeepSpeed', 'FlexSP', 'Arachne']

AESTHETICS = {
    'color_map': {
        'Megatron-LM': '#ACD7E5',
        'DeepSpeed':   '#EFD2BA',
        'FlexSP':      '#59a14f',
        'Arachne':     '#FE8B01'
    },
    'hatch_map': {
        'Megatron-LM': '/',
        'DeepSpeed':   '..',
        'FlexSP':      '||',
        'Arachne':     None
    }
}

# ======================================================================
# ======================================================================

PLOT_VALUE_OVERRIDES = {}

def apply_plot_value_override(plot_data, subplot_title):
    if subplot_title not in PLOT_VALUE_OVERRIDES:
        return plot_data

    override_cfg = PLOT_VALUE_OVERRIDES[subplot_title]
    new_data = {}

    for case, vals in plot_data.items():
        new_vals = vals.copy()
        if case in override_cfg:
            for sys, v in override_cfg[case].items():
                if sys in new_vals:
                    print(
                        f"   [OVERRIDE] [{subplot_title}] "
                        f"{case} {sys}: {new_vals[sys]:.4f} -> {v:.4f}"
                    )
                    new_vals[sys] = v
        new_data[case] = new_vals

    return new_data

def build_experiment_path(
    base_dir,
    system_folder,
    base_model,
    resolution,
    filename,
    max_frames=None
):
    if max_frames is not None:
        return os.path.join(
            base_dir, system_folder, base_model, resolution, max_frames, filename
        )
    else:
        return os.path.join(
            base_dir, system_folder, base_model, resolution, filename
        )

def dump_iteration_level_throughput(
    case_name,
    common_iters,
    frames,
    times_by_system,
    gpu_count,
    output_dir="iteration_throughput_debug"
):
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    records = []
    per_system_time_dict = {sys: {} for sys in times_by_system.keys()}

    for it in common_iters:
        row = {
            "iteration": it,
            "frames": frames[it],
        }

        for sys, times in times_by_system.items():
            t = times.get(it, None)
            row[f"{sys}_time"] = t
            per_system_time_dict[sys][it] = t

        for sys, times in times_by_system.items():
            t = times.get(it, None)
            if t is not None and t > 0:
                row[f"{sys}_throughput"] = frames[it] / (gpu_count * t)
            else:
                row[f"{sys}_throughput"] = None

        if "Megatron-LM" in times_by_system:
            ref_tp = row.get("Megatron-LM_throughput")
            for sys in times_by_system:
                cur_tp = row.get(f"{sys}_throughput")
                if ref_tp and cur_tp:
                    row[f"{sys}_vs_megatron"] = cur_tp / ref_tp
                else:
                    row[f"{sys}_vs_megatron"] = None

        records.append(row)

    with open(os.path.join(case_dir, "iteration_level_summary.json"), "w") as f:
        json.dump(records, f, indent=2)

    for sys, time_dict in per_system_time_dict.items():
        with open(os.path.join(case_dir, f"{sys}_times.json"), "w") as f:
            json.dump(time_dict, f, indent=2)

def load_iteration_times(system, base_model, resolution, max_frames=None):
    if system == 'Arachne':
        base_dir = 'dynamic_flex_exp_log'
        system_folder = 'arachne'
        filename = 'figures/iteration_times_summary.json'
    else:
        base_dir = 'baseline_log'
        if system == 'Megatron-LM':
            system_folder = 'megatron-lm'
            filename = 'figures/iteration_times_summary.json'
        elif system == 'DeepSpeed':
            system_folder = 'deepspeed'
            filename = 'iteration_times_summary.json'
        elif system == 'FlexSP':
            system_folder = 'flex_sp'
            filename = 'figures/iteration_times_summary.json'
        else:
            return None, None

    path = build_experiment_path(
        base_dir, system_folder, base_model, resolution, filename, max_frames
    )

    try:
        with open(path, 'r', encoding="utf-8-sig") as f:
            raw = json.load(f)
            data = {int(k): v for k, v in raw.items()}
        return data, path
    except FileNotFoundError:
        return None, path

def load_total_frames(base_model, resolution, max_frames=None):
    if max_frames is not None:
        filename = f"total_frames_{base_model}_{resolution}_{max_frames}.txt"
    else:
        filename = f"total_frames_{base_model}_{resolution}.txt"

    path = os.path.join("simulation_data", filename)
    frames = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                m = re.search(r'iteration\s+(\d+)\s*:\s*(\d+)', line)
                if m:
                    frames[int(m.group(1))] = int(m.group(2))
        return frames
    except FileNotFoundError:
        return None

# ==============================================================================
# ==============================================================================

def plot_single_subplot(ax, throughput_data, annotations, title, tick_map=None):
    case_keys = list(throughput_data.keys())
    xticks = [tick_map.get(k, k) for k in case_keys] if tick_map else case_keys

    systems = [s for s in SYSTEMS_ORDER
               if any(s in throughput_data[c] for c in case_keys)]

    x = np.arange(len(case_keys))
    width = 0.75 / len(systems)

    for i, sys in enumerate(systems):
        offset = width * (i - (len(systems) - 1) / 2)
        values = [throughput_data[c].get(sys, 0) for c in case_keys]

        bars = ax.bar(
            x + offset, values, width,
            color=AESTHETICS['color_map'][sys],
            hatch=AESTHETICS['hatch_map'][sys],
            edgecolor='black',
            linewidth=2.2 if sys == 'Arachne' else 0.8,
            zorder=3,
            label=sys
        )

        for j, bar in enumerate(bars):
            h = bar.get_height()
            if h <= 0:
                continue

            ax.annotate(f"{h:.2f}",
                        (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', fontsize=8, fontweight='bold')

            if sys == 'Arachne' and case_keys[j] in annotations:
                ann = annotations[case_keys[j]]
                texts = []
                for k in ['vs_megatron', 'vs_deepspeed', 'vs_flexsp']:
                    if k in ann:
                        texts.append(rf"$\mathbf{{{ann[k]:.2f}\times}}$")
                if texts:
                    ax.annotate(", ".join(texts),
                                (bar.get_x() + bar.get_width()/2, h),
                                xytext=(0, 22), textcoords="offset points",
                                ha='center', fontsize=9,
                                fontweight='bold', color='red')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.tick_params(axis='y')
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_ylim(0, max(v for c in throughput_data.values() for v in c.values()) * 1.35)

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    final_data = {}

    print("=" * 60)
    print(" Computing Throughput (FlexSP optional, max_frames optional)")
    print("=" * 60)

    for case, cfg in THROUGHPUT_CONFIG.items():
        print(f"\n--- {case} ---")

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        gpu_cnt = 16
        if 'nodes' in case:
            m = re.search(r'(\d+)', case)
            if m:
                gpu_cnt = int(m.group(1)) * 8

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        max_frames = cfg.get('max_frames', None)

        # ------------------------------------------------------------
        #    - old: total_frames_{base_model}_{resolution}.txt
        #    - new: total_frames_{base_model}_{resolution}_{max_frames}.txt
        # ------------------------------------------------------------
        frames = load_total_frames(
            base_model=cfg['base_model_name'],
            resolution=cfg['resolution'],
            max_frames=max_frames
        )
        if not frames:
            print(f"   [SKIP] frames not found for {case}")
            continue
        frame_keys = sorted(frames.keys())
        print(f"   [FRAMES] iters={len(frame_keys)}")
        if frame_keys:
            print(f"            min/max: {frame_keys[0]} / {frame_keys[-1]}")

            gaps = []
            for a, b in zip(frame_keys, frame_keys[1:]):
                if b != a + 1:
                    gaps.append((a, b))
            if gaps:
                print(f"            ❌ non-contiguous gaps (show up to 10): {gaps[:10]}")
            else:
                print(f"            ✅ keys are contiguous")
        # ------------------------------------------------------------
        #    - old: .../<base_model>/<resolution>/...
        #    - new: .../<base_model>/<resolution>/<max_frames>/...
        # ------------------------------------------------------------
       # ------------------------------------------------------------
        # ------------------------------------------------------------
        times = {}

        for sys in SYSTEMS_ORDER:
            t, p = load_iteration_times(
                system=sys,
                base_model=cfg['base_model_name'],
                resolution=cfg['resolution'],
                max_frames=max_frames
            )
            if t:
                times[sys] = t

        print(f"   [SUMMARY] Loaded systems for {case}:")
        for s, td in times.items():
            print(f"     - {s:<10}: {len(td)} iterations")


        if not times:
            print(f"   [SKIP] no system times loaded for {case}")
            continue

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        common_iters = set.intersection(
            *[set(times[s].keys()) & set(frames.keys()) for s in times]
        )
        if not common_iters:
            print(f"   [SKIP] no common iterations for {case}")
            continue

        common_iters = sorted(common_iters)
        print(f"   -> common iters: {len(common_iters)}")

        common_set = set(common_iters)
        for sys, td in times.items():
            sys_set = set(td.keys()) & set(frames.keys())
            missing = sorted(common_set ^ sys_set)
            if missing:
                print(f"   [DEBUG] {sys} differs from common:")
                print(f"           diff count = {len(missing)}")
                print(f"           examples   = {missing[:10]}")

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        dump_iteration_level_throughput(
            case_name=case,
            common_iters=common_iters,
            frames=frames,
            times_by_system=times,
            gpu_count=gpu_cnt
        )

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        case_tp = {}
        total_frames = sum(frames[i] for i in common_iters)

        for sys in times:
            total_time = sum(times[sys][i] for i in common_iters if times[sys][i] > 0)
            if total_time <= 0:
                continue
            case_tp[sys] = total_frames / (gpu_cnt * total_time)

        if case_tp:
            final_data[case] = case_tp

   # =========================
    # Speedup annotations（Arachne vs baselines）
    # =========================
    annotations = {}
    print("\n" + "=" * 60)
    print(" Case-level Throughput & Speedup Summary")
    print("=" * 60)

    for case, vals in final_data.items():
        if 'Arachne' not in vals:
            continue

        print(f"\n[CASE] {case}")
        print("  Throughput (Frames / GPU / sec):")

        for sys in SYSTEMS_ORDER:
            if sys in vals:
                print(f"    - {sys:<10}: {vals[sys]:.4f}")

        ann = {}
        for base, key in [('Megatron-LM', 'vs_megatron'),
                        ('DeepSpeed', 'vs_deepspeed'),
                        ('FlexSP', 'vs_flexsp')]:
            if base in vals and vals[base] > 0:
                speedup = vals['Arachne'] / vals[base]
                ann[key] = speedup
                print(f"    -> Speedup (Arachne vs {base}): {speedup:.2f}x")

        if ann:
            annotations[case] = ann


    # =========================
    # Plot
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # ------------------------------------------------------------
    # Subplot 1: Model Size Scalability
    # ------------------------------------------------------------
    title = "Model Size Scalability"
    data_1 = {
        k: final_data[k]
        for k in ['Wan-1.3B', 'CogVideoX-5B', 'HunyuanVideo-13B']
        if k in final_data
    }
    data_1 = apply_plot_value_override(data_1, title)

    plot_single_subplot(
        axes[0],
        data_1,
        annotations,
        title,
        {'Wan-1.3B': 'Wan2.1', 'CogVideoX-5B': 'CogVideoX', 'HunyuanVideo-13B': 'HunyuanVideo'}
    )

    axes[0].set_xlabel("Model Type", fontsize=16, fontweight='bold')
    # ------------------------------------------------------------
    # Subplot 2: Workload Heterogeneity
    # ------------------------------------------------------------
    title = "Workload Heterogeneity"
    data_2 = {
        k: final_data[k]
        for k in ['hunyuan-53', 'hunyuan-81', 'hunyuan-105-frames']
        if k in final_data
    }
    data_2 = apply_plot_value_override(data_2, title)

    plot_single_subplot(
        axes[1],
        data_2,
        annotations,
        title,
        {'hunyuan-53': '53 frames', 'hunyuan-81': '81 frames', 'hunyuan-105-frames': '105 frames'}
    )
    axes[1].set_xlabel("Max Frames", fontsize=16, fontweight='bold')
    # ------------------------------------------------------------
    # Subplot 3: Cluster Size Scalability
    # ------------------------------------------------------------
    title = "Cluster Size Scalability"
    data_3 = {
        k: final_data[k]
        for k in ['hunyuan-2nodes', 'hunyuan-4nodes', 'hunyuan-8nodes']
        if k in final_data
    }
    data_3 = apply_plot_value_override(data_3, title)

    plot_single_subplot(
        axes[2],
        data_3,
        annotations,
        title,
        {'hunyuan-2nodes': '16', 'hunyuan-4nodes': '32', 'hunyuan-8nodes': '64'}
    )

    axes[0].set_ylabel("Throughput (Frames/GPU/Second)", fontsize=14, fontweight='bold')

    axes[2].set_xlabel("#GPUs", fontsize=16, fontweight='bold')
    present_systems = [s for s in SYSTEMS_ORDER
                       if any(s in v for v in final_data.values())]

    handles = [
        Patch(facecolor=AESTHETICS['color_map'][s],
              hatch=AESTHETICS['hatch_map'][s],
              edgecolor='black',
              linewidth=2.2 if s == 'Arachne' else 1.0,
              label=s)
        for s in present_systems
    ]

    fig.legend(handles, present_systems,
               loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(present_systems), frameon=False, fontsize=17)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("throughput_comparison_with_optional_flexsp.pdf", bbox_inches='tight')
    plt.show()

    print("\n[SUCCESS] Saved: throughput_comparison_with_optional_flexsp.pdf")
