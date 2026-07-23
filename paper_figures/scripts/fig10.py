import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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
# 3. Configuration
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
    json_path = os.path.join(
        BASE_DIR, m["log_root"], m["framework"],
        MODEL, RESOLUTION, "figures", "flops",
        f"iteration_{iteration_id:03d}_flops_simple.json"
    )
    data = try_load_iteration_flops(json_path)

    # Fail loudly if the required data is missing (no mock fallback):
    # the figure must only ever be rendered from real measured/derived data.
    if data is None:
        raise FileNotFoundError(
            f"Required FLOPs data not found for {m['name']}: {json_path}\n"
            f"Generate it first via: python -m MFU.task_yaml_cal_mfu_flops ..."
        )

    loaded_methods.append((m, data))

# ============================================================
# 5. Plotting
# ============================================================
num_plots = len(loaded_methods)
fig, axes = plt.subplots(1, num_plots, sharey=True)
if num_plots == 1:
    axes = [axes]

legend_handles = []
legend_labels = []

for i, (ax, (method, data)) in enumerate(zip(axes, loaded_methods)):
    ranks, flops, metrics = data
    bottleneck = flops.max()
    bottleneck_rank = int(ranks[np.argmax(flops)])
    bottleneck_value = bottleneck
    gap = bottleneck - flops

    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Bars
    ax.bar(
        ranks, flops,
        color="#f28e2b",
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    ax.bar(
        ranks, gap, bottom=flops,
        facecolor="none",
        hatch="////",
        edgecolor="#e15759",
        linewidth=1.0,
        zorder=3
    )

    ax.bar(
        ranks, gap, bottom=flops,
        color="none",
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    # Bottleneck line
    ax.axhline(
        bottleneck,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        zorder=4
    )

    # ---- Bottleneck annotation (NEW FEATURE) ----
    # annot_text = (
    #     f"Bottleneck Rank: {bottleneck_rank}\n"
    #     f"Assigned: {bottleneck_value:.1f} TFLOPS"
    # )

    # ax.annotate(
    #     annot_text,
    #     xy=(bottleneck_rank, bottleneck_value),
    #     xycoords="data",
    #     textcoords="axes fraction",
    #     arrowprops=dict(
    #         arrowstyle="->",
    #         color="#d62728",
    #         linewidth=1.8
    #     ),
    #     bbox=dict(
    #         boxstyle="round,pad=0.4",
    #         facecolor="white",
    #         edgecolor="#d62728",
    #         alpha=0.95
    #     ),
    #     fontsize=13,
    #     ha="left",
    #     va="top",
    #     zorder=6,
    # )

    if i == 0:
        assigned_handle = mpatches.Patch(
            # facecolor="#bdbdbd",
            facecolor="#f28e2b",
            edgecolor="black",
            label="Assigned Workload"
        )
        gap_handle = mpatches.Patch(
            facecolor="none",
            edgecolor="#e15759",
            hatch="////",
            label="Imbalance Gap"
        )
        bottleneck_handle = mlines.Line2D(
            [], [], color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Bottleneck"
        )

        legend_handles.extend([assigned_handle, gap_handle, bottleneck_handle])
        legend_labels.extend(["Assigned Workload", "Imbalance Gap", "Bottleneck"])

    ax.set_title(method["title"], pad=12, fontweight="bold")
    ax.set_xlabel("Rank ID", fontweight="bold")

    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15])
    ax.set_xlim(-1, 16)

    text_str = (
        f"$\\mathbf{{CV}} = {metrics.get('coefficient_of_variation', 0):.3f}$\n"
        f"Max/Min = {metrics.get('max_min_ratio', 1):.2f}"
    )
    if i == 1:
        text_x = 0.625
        # text_ha = "right"
    else:
        text_x = 0.05
        # text_ha = "left"

    ax.text(
        text_x, 0.95,
        text_str,
        transform=ax.transAxes,
        verticalalignment="top",
        # horizontalalignment=text_ha,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray",
            linewidth=0.5
        ),
        fontsize=14,
        zorder=5
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel("TFLOPS", fontweight="bold")

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
