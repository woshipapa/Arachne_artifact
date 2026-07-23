
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# MAIN INPUT JSON PATHS (one per method):
#   ./{log_root}/{framework}/{MODEL}/{RESOLUTION}/figures/flops/iteration_{iteration_id:03d}_flops_simple.json
# What this script does:
#   1) Load per-rank FLOPs + metrics for each baseline (or mock data if missing).
#   2) Plot per-rank bars, imbalance gaps, and the bottleneck line.
#   3) Annotate CV and max/min ratio, then save a PDF figure.
# Expected JSON contents:
#   {
#     "ranks": { "0": <flops>, "1": <flops>, ... },
#     "metrics": {
#       "coefficient_of_variation": <float>,
#       "max_min_ratio": <float>
#     }
#   }

# ============================================================
# ============================================================
LEGEND_Y_POS = 0.90
SUBPLOT_TOP  = 0.85

# ============================================================
# 1. Paper-style plotting config
# ============================================================
from paper_plot_style import apply_paper_style
apply_paper_style(figsize=(16, 5.5))
# ============================================================
# 2. Data Loader
# ============================================================
def try_load_iteration_flops(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    ranks = np.array(sorted(int(r) for r in data["ranks"].keys()))
    flops = np.array([data["ranks"][str(r)] for r in ranks]) / 1e12
    metrics = data.get("metrics", {})
    return ranks, flops, metrics

# ============================================================
# ============================================================
iteration_id = 29
BASE_DIR = "."
MODEL = "hunyuan-129"
RESOLUTION = "720p"

METHODS = [
    {
        "name": "Megatron-LM", 
        "log_root": "baseline_log",       
        "framework": "megatron-lm",  
        "color": "#f28e2b", 
        "title": "(a) Megatron-LM"
    },
    {
        "name": "FlexSP",      
        "log_root": "baseline_log",       
        "framework": "flex_sp",      
        "color": "#4e79a7", 
        "title": "(b) FlexSP"
    },
    {
        "name": "Arachne",     
        "log_root": "dynamic_flex_exp_log",
        "framework": "arachne",     
        "color": "#59a14f", 
        "title": "(c) Arachne"
    },
]

# ============================================================
# 4. Load Data
# ============================================================
loaded_methods = []
for m in METHODS:
    json_path = os.path.join(BASE_DIR, m["log_root"], m["framework"], MODEL, RESOLUTION, "figures", "flops", f"iteration_{iteration_id:03d}_flops_simple.json")
    data = try_load_iteration_flops(json_path)
    
    # --- MOCK DATA FOR DEMO ---
    if data is None:
        ranks = np.arange(16)
        if m['name'] == 'Arachne': flops = np.random.normal(100, 2, 16)
        else: flops = np.random.normal(100, 15, 16); flops[0]=140
        metrics = {'coefficient_of_variation': np.std(flops)/np.mean(flops), 'max_min_ratio': np.max(flops)/np.min(flops)}
        data = (ranks, flops, metrics)
    # --------------------------
    
    if data is not None:
        loaded_methods.append((m, data))

# ============================================================
# 5. Plotting
# ============================================================
num_plots = len(loaded_methods)
fig, axes = plt.subplots(1, num_plots, sharey=True)
if num_plots == 1: axes = [axes]

legend_handles = []
legend_labels = []

for i, (ax, (method, data)) in enumerate(zip(axes, loaded_methods)):
    ranks, flops, metrics = data
    bottleneck = flops.max()
    gap = bottleneck - flops

    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Bars
    bar1 = ax.bar(ranks, flops, color=method["color"], edgecolor="black", linewidth=0.8, label="Assigned FLOPs", zorder=3)
    bar2 = ax.bar(
    ranks, gap, bottom=flops,
    facecolor="none",
    hatch="////",
    edgecolor="#e15759",
    linewidth=1.0,
    label="Imbalance Gap",
    zorder=3,
)

    ax.bar(ranks, gap, bottom=flops, color="none", edgecolor="black", linewidth=0.8, zorder=3)

    # Line
    line = ax.axhline(bottleneck, color="#d62728", linestyle="--", linewidth=1.5, label="Bottleneck", zorder=4)

    if i == 0:
        legend_handles.extend([bar1, bar2, line])
        legend_labels.extend(["Assigned Workload", "Imbalance Gap", "Bottleneck"])

    ax.set_title(method["title"], pad=12, fontweight='bold')
    ax.set_xlabel("Rank ID", fontweight = "bold")
    
    ax.set_xticks([0, 5, 10, 15]) 
    ax.set_xticklabels([0, 5, 10, 15]) 
    ax.set_xlim(-1, 16)

    # Metrics Text
    text_str = f"$\\mathbf{{CV}} = {metrics.get('coefficient_of_variation', 0):.3f}$\nMax/Min = {metrics.get('max_min_ratio', 1):.2f}"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5), fontsize=14, zorder=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel("TFLOPS", fontweight = "bold")

# ============================================================
# 6. Layout & Save
# ============================================================
plt.tight_layout(rect=[0, 0, 1, SUBPLOT_TOP]) 

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, LEGEND_Y_POS), 
    ncol=3,
    frameon=False,
    fontsize=16
)

save_path = f"per_rank_flops_balance_iteration_{iteration_id:03d}_3methods.pdf"
plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
