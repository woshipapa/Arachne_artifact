
import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Style
# =========================
try:
    from paper_plot_style import apply_paper_style
    apply_paper_style(figsize=(7.1, 5.3))
except ImportError:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Verdana']})

# =========================
# 1. Config
# =========================
INPUT_JSONL = "pareto_genetic_32gpus.jsonl"
MAX_GENERATION = 150

MEGATRON_MAKESPAN = 50.78
FLEXSP_MAKESPAN = 41.77

BEST_KNOWN_MAKESPAN = 33.454
SOLVER_TIME_LIMIT = 600

TARGET_RATIO = 0.90
ANNOTATION_TEXT_OFFSET = (5, 2.0)

# =========================
# 2. Load Data
# =========================
all_iters = []
with open(INPUT_JSONL, "r") as f:
    for line in f:
        if line.strip():
            all_iters.append(json.loads(line)["pareto"])

gens = np.arange(0, MAX_GENERATION + 1)

aligned, time_aligned = [], []
for pareto in all_iters:
    pareto = sorted(pareto, key=lambda x: x["generation"])
    curve, time_curve = {}, {}
    best = np.inf
    for p in pareto:
        best = min(best, p["best_makespan"])
        curve[p["generation"]] = best
        time_curve[p["generation"]] = p["elapsed_plan_time"]

    vals, times = [], []
    vals.append(curve.get(1, np.inf))
    times.append(0.0)

    last_best, last_time = vals[0], 0.0
    for g in range(1, MAX_GENERATION + 1):
        if g in curve:
            last_best = curve[g]
            last_time = time_curve[g]
        vals.append(last_best)
        times.append(last_time)

    aligned.append(vals)
    time_aligned.append(times)

aligned = np.array(aligned)
time_aligned = np.array(time_aligned)

median = np.median(aligned, axis=0)
median_time = np.median(time_aligned, axis=0)

# =========================
# 3. Target
# =========================
target_makespan = BEST_KNOWN_MAKESPAN / TARGET_RATIO

found = False
idx = np.where(median <= target_makespan)[0]
if len(idx) > 0:
    found = True
    idx = idx[0]
    g_target = gens[idx]
    t_target = median_time[idx]
    val_target = median[idx]

# =========================
# 4. Plot
# =========================
fig, ax = plt.subplots()
# ax.set_ylim(33.0, 52.0)


ymin, ymax = 32.0, 44.0
ax.set_ylim(ymin, ymax)

ax.set_yticks(np.arange(32, 45, 4))  # 32, 36, 40, 44

# Arachne curve
ax.plot(
    gens, median,
    linewidth=2.5,
    color='#D62728',
    label="Arachne",
    zorder=5
)

# Baselines
# ax.axhline(MEGATRON_MAKESPAN, linestyle=":", linewidth=2, color='gray', label="Megatron-LM")
# ax.axhline(FLEXSP_MAKESPAN, linestyle="--", linewidth=2, color='tab:blue', label="FlexSP")
ax.axhline(
    BEST_KNOWN_MAKESPAN,
    linestyle="-.",
    linewidth=2,
    color='black',
    label=f"Best Known (Solver {SOLVER_TIME_LIMIT}s)"
)

# Annotation
if found:
    ax.axvline(g_target, linestyle="--", linewidth=1.5, color='gray', alpha=0.8)
    ax.scatter([g_target], [val_target], color='black', s=50, zorder=6)

    pct = int(TARGET_RATIO * 100)
    # ax.annotate(
    #     f"Reaches {pct}% of Best Known\n"
    #     f"≈ {g_target} gens\n"
    #     f"≈ {t_target:.1f} s",
    #     xy=(g_target, val_target),
    #     xytext=(g_target + ANNOTATION_TEXT_OFFSET[0],
    #             val_target + ANNOTATION_TEXT_OFFSET[1]),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    #     # fontsize=16,
    #     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    #     verticalalignment='bottom'
    # )
    ax.annotate(
    f"Reaches {pct}% of Best Known\n"
    f"≈ {g_target} gens\n"
    f"≈ {t_target:.1f} s",
    xy=(g_target, val_target),
    xytext=(g_target + ANNOTATION_TEXT_OFFSET[0],
            val_target + ANNOTATION_TEXT_OFFSET[1]),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.6,
        connectionstyle="arc3,rad=0.2",
        color="gray"
    ),
    fontsize=15,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.4",
        fc="white",
        ec="gray",
        linewidth=1.6,
        alpha=0.95
    ),
    verticalalignment='bottom'
)


# =========================
# 5. Formatting
# =========================
ax.set_xlabel("Planning Budget (Generation)", fontweight="bold")
ax.set_ylabel("Best Makespan (s)", fontweight="bold")

ax.set_xlim(left=0)
ax.margins(x=0)
ax.tick_params(axis='both', which='major', width=1.5)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(True, linestyle='--', alpha=0.6)

# ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.21),
#     ncol=2,
#     frameon=False,
#     fontsize=16
# )

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    frameon=False,
    fontsize=16
)


# =========================
# Save
# =========================
fig.savefig(
    f"anytime_bestknown_{int(TARGET_RATIO*100)}pct_32gpu.pdf",
    dpi=300
)
plt.show()
