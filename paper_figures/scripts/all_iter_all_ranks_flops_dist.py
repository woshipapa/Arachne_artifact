import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (7, 4)
})


# ============================================================
# ============================================================
def collect_per_rank_flops(flops_dir, iterations=None):
    """
    flops_dir:
      .../figures/flops/

    iterations:

      np.ndarray, shape = [len(iterations) * num_ranks]
    """
    flops_all = []

    if iterations is None:
        json_files = sorted(
            glob.glob(os.path.join(flops_dir, "iteration_*_flops_simple.json"))
        )
    else:
        json_files = [
            os.path.join(
                flops_dir,
                f"iteration_{int(it):03d}_flops_simple.json"
            )
            for it in iterations
        ]

    if not json_files:
        raise RuntimeError(f"No FLOPs json found in {flops_dir}")

    for jf in json_files:
        if not os.path.exists(jf):
            raise FileNotFoundError(f"Missing FLOPs file: {jf}")

        with open(jf, "r") as f:
            data = json.load(f)

        for _, flops in data["ranks"].items():
            flops_all.append(flops / 1e12)  # -> TFLOPs

    return np.array(flops_all)


# ============================================================
# ============================================================
BASE_DIR = "."
MODEL = "hunyuan-129"
RESOLUTION = "720p"

BASELINE = "flex_sp"
OURS = "arachne"

ITERATIONS = [
    0,
    12,
    29,
    3,
    ]
baseline_flops_dir = os.path.join(
    BASE_DIR,
    "baseline_log",
    BASELINE,
    MODEL,
    RESOLUTION,
    "figures",
    "flops",
)

ours_flops_dir = os.path.join(
    BASE_DIR,
    "dynamic_flex_exp_log",
    OURS,
    MODEL,
    RESOLUTION,
    "figures",
    "flops",
)

# ============================================================
# ============================================================
baseline_flops = collect_per_rank_flops(
    baseline_flops_dir,
    iterations=ITERATIONS
)

ours_flops = collect_per_rank_flops(
    ours_flops_dir,
    iterations=ITERATIONS
)


# ============================================================
# ============================================================
def coefficient_of_variation(x):
    return np.std(x) / np.mean(x)

cv_baseline = coefficient_of_variation(baseline_flops)
cv_ours = coefficient_of_variation(ours_flops)


# ============================================================
# ============================================================
fig, ax = plt.subplots()

parts = ax.violinplot(
    [baseline_flops, ours_flops],
    positions=[1, 2],
    widths=0.7,
    showmeans=True,
    showextrema=False
)

colors = ['#4e79a7', '#59a14f']  # baseline / ours
for body, color in zip(parts['bodies'], colors):
    body.set_facecolor(color)
    body.set_alpha(0.85)
    body.set_edgecolor('black')
    body.set_linewidth(0.8)

parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(2)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Baseline', 'Ours'])
ax.set_ylabel('Per-rank FLOPs (TFLOPs)')
ax.set_title('Per-rank FLOPs Distribution Across Iterations')

ax.grid(axis='y', linestyle='--', alpha=0.3)

# ============================================================
# ============================================================
ax.text(
    0.02, 0.96,
    f"CV (Baseline) = {cv_baseline:.3f}\n"
    f"CV (Ours) = {cv_ours:.3f}",
    transform=ax.transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)

# ============================================================
# ============================================================
plt.tight_layout()
plt.savefig(
    "per_rank_flops_distribution_across_iterations_violin.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
