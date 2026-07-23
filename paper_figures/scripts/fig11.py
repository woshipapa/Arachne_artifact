import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

from paper_plot_style import apply_paper_style
apply_paper_style(figsize=(8, 5))

# ============================================================
# ============================================================
def coefficient_of_variation(x: np.ndarray) -> float:
    mean = np.mean(x)
    if mean == 0:
        return float("nan")
    return float(np.std(x) / mean)

def collect_iteration_cv_map(flops_dir, iterations):
    cv_map = {}
    for it in iterations:
        json_path = os.path.join(flops_dir, f"iteration_{int(it):03d}_flops_simple.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        flops = np.array(list(data["ranks"].values()), dtype=np.float64)
        if flops.size == 0:
            continue
        cv = coefficient_of_variation(flops)
        if not np.isnan(cv):
            cv_map[it] = cv
    return cv_map


def moving_average(x, window=5):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def moving_average_same(x, window=5):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")

# ============================================================
# 3. Phase Configuration (NEW)
# ============================================================
BASE_DIR = "."
GLOBAL_PHASE_SHUFFLE_SEED = 20260113 
PHASES = [
    {
        "model": "hunyuan-129",
        "resolution": "720p",
        "iterations": [    0,
    12,
    29,
    3,
    25,
    2,
    23,
    1,
    10,
    26,
    5,
    15,
    24,
    14,
    7,
    28,
    9,
    17,
    20,
    19,
    4,
    22,],
    },
    # {
    #     "model": "hunyuan",
    #     "max_frames": "105",
    #     "resolution": "720p",
    #     "iterations": [    30,
    # 23,
    # 18,
    # 25,
    # 40,
    # 19,
    # 33,
    # 36,
    # 34,
    # 6,
    # 39,
    # 41,
    # 5,
    # 29,
    # 38,
    # 1,
    # 20,
    # 17,],
    # },

    {
        "model": "hunyuan-129",
        # "max_frames": "210",
        "resolution": "1080p",
        "iterations": [     25,
    7,
    11,
    4,
    22,
    32,
    24,
    0,
    15,
    21,
    29,
    34,
    5,
    30,
    8,
    26,
    14,
    3,
    19,
    31,
    2,
    33,
    36,
    28,
    23,
    9,
    17,
    16,
    18,
    6,
    10,]
    }
]

# ============================================================
# ============================================================
METHODS = [
    {
        "name": "Megatron-LM",
        "label": "Megatron-LM",
        "log_root": "baseline_log",
        "framework": "megatron-lm",
        "color": "#f28e2b",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.5,
    },
    {
        "name": "FlexSP",
        "label": "FlexSP",
        "log_root": "baseline_log",
        "framework": "flex_sp",
        "color": "#4e79a7",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 1.5,
    },
    {
        "name": "Arachne",
        "label": "Arachne",
        "log_root": "dynamic_flex_exp_log",
        "framework": "arachne",
        "color": "#59a14f",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.5,
    },
]

# ============================================================
# ============================================================
method_cv_series = {m["name"]: [] for m in METHODS}

for phase_idx, phase in enumerate(PHASES):
    model = phase["model"]
    resolution = phase["resolution"]

    # ---- reproducible per-phase shuffle ----
    iterations = list(phase["iterations"])
    phase_seed = GLOBAL_PHASE_SHUFFLE_SEED + phase_idx
    rng_phase = random.Random(phase_seed)
    rng_phase.shuffle(iterations)

    print(
        f"[Phase {phase_idx}] {model}-{resolution} "
        f"shuffled iterations (seed={phase_seed}): {iterations}"
    )

    phase_cv_maps = {}

    for m in METHODS:
        path_parts = [
        BASE_DIR,
        m["log_root"],
        m["framework"],
        model,
        resolution,
        ]

        if "max_frames" in phase and phase["max_frames"] is not None:
            path_parts.append(str(phase["max_frames"]))

        path_parts.extend(["figures", "flops"])

        flops_dir = os.path.join(*path_parts)
        cv_map = collect_iteration_cv_map(flops_dir, iterations)

        # Fail loudly if the required data is missing (no mock fallback):
        # the figure must only ever be rendered from real measured/derived data.
        if not cv_map:
            raise FileNotFoundError(
                f"No FLOPs CV data found for {m['name']} @ {model}-{resolution} "
                f"under {flops_dir}. Generate it first via: "
                f"python -m MFU.task_yaml_cal_mfu_flops ..."
            )

        phase_cv_maps[m["name"]] = cv_map

    common_iters = set(iterations)
    for cv_map in phase_cv_maps.values():
        common_iters &= set(cv_map.keys())
    ordered_iters = [it for it in iterations if it in common_iters]

    for m in METHODS:
        name = m["name"]
        method_cv_series[name].extend(
            phase_cv_maps[name][it] for it in ordered_iters
        )

# sanity check
if not any(len(v) > 0 for v in method_cv_series.values()):
    raise RuntimeError("No valid CV data collected.")

# ============================================================
# ============================================================
fig, ax = plt.subplots()

total_len = len(next(iter(method_cv_series.values())))
x_indices = np.arange(1, total_len + 1)

for m in METHODS:
    name = m["name"]
    cvs = method_cv_series[name]
    ax.plot(
    x_indices,
    cvs,
    label=m["label"],
    color=m["color"],
    linestyle=m["linestyle"],
    marker=m["marker"],
    linewidth=m["linewidth"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=1.0,
    alpha=0.9,
)


    # window = 5
    # smooth_cvs = moving_average_same(cvs, window=window)
    # ax.plot(
    #     x_indices, smooth_cvs,
    #     label=m["label"],
    #     color=m["color"],
    #     linestyle=m["linestyle"],
    #     linewidth=m["linewidth"],
    #     alpha=0.95,
    # )


ax.set_xlabel("Consecutive Training Iterations", fontweight="bold")
ax.set_ylabel("Coefficient of Variation (CV)", fontweight="bold")

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.set_xlim(0.5, total_len + 0.5)

ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=True,
    fancybox=False,
    edgecolor="gray",
    framealpha=0.9,
    loc="upper right",
)

ax.text(
    0.02,
    0.95,
    r"Lower is better $\downarrow$",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    verticalalignment="top",
)

plt.tight_layout()
plt.savefig("cv_stability_over_iterations.pdf", dpi=300, bbox_inches="tight")
plt.show()
