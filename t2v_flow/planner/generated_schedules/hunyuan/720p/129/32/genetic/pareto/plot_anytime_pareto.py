import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Unified paper style
# =========================
from prof_results.paper_plot_style import apply_paper_style
apply_paper_style(figsize=(8, 5))

# =========================
# Config
# =========================
INPUT_JSONL = "pareto_genetic_32gpus.jsonl"
MAX_GENERATION = 150
ANNOTATION_Y_OFFSET = 2.0

MEGATRON_MAKESPAN = 50.78
FLEXSP_MAKESPAN = 41.77
BEST_KNOWN_MAKESPAN = 29.07   # ← 32 GPUs 下 best-known（例如 1800s 后）

# =========================
# Load data
# =========================
all_iters = []

with open(INPUT_JSONL, "r") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        all_iters.append(record["pareto"])

gens = np.arange(0, MAX_GENERATION + 1)

# =========================
# Build aligned best-so-far curves
# =========================
aligned = []
time_aligned = []

for pareto in all_iters:
    pareto = sorted(pareto, key=lambda x: x["generation"])

    best = np.inf
    curve = {}
    time_curve = {}

    for p in pareto:
        best = min(best, p["best_makespan"])
        curve[p["generation"]] = best
        time_curve[p["generation"]] = p["elapsed_plan_time"]

    vals, times = [], []
    last_best, last_time = np.inf, 0.0

    # generation 0
    vals.append(curve.get(1, np.inf))
    times.append(0.0)

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

# =========================
# Median curve
# =========================
median = np.median(aligned, axis=0)
median_time = np.median(time_aligned, axis=0)

# =========================
# 80% improvement point
# =========================
M0 = median[0]
M_final = median[-1]
target = M0 - 0.8 * (M0 - M_final)

g_80 = next(g for g, m in zip(gens, median) if m <= target)
t_80 = median_time[g_80]

print(f"80% improvement @ gen {g_80}, ~{t_80:.1f}s")

# =========================
# Plot
# =========================
fig, ax = plt.subplots()

# ---- Arachne anytime curve ----
ax.plot(
    gens,
    median,
    linewidth=2.4,
    label="Arachne",
    zorder=4,
)

# ---- Best-known line ----
ax.axhline(
    BEST_KNOWN_MAKESPAN,
    color="black",
    linewidth=2.2,
    linestyle="-",
    label="Best-known (oracle)",
    zorder=3,
)

# ---- Baselines ----
ax.axhline(
    MEGATRON_MAKESPAN,
    linestyle=":",
    linewidth=2.0,
    label="Megatron-LM",
    zorder=2,
)

ax.axhline(
    FLEXSP_MAKESPAN,
    linestyle="--",
    linewidth=2.0,
    label="FlexSP",
    zorder=2,
)

# ---- 80% vertical line ----
ax.axvline(
    g_80,
    linestyle="--",
    linewidth=1.6,
    color="gray",
    zorder=1,
)

ax.scatter(g_80, median[g_80], zorder=5)

# =========================
# Annotation
# =========================
ax.text(
    g_80 + 4,
    median[g_80] + ANNOTATION_Y_OFFSET,
    f"80% improvement\n≈ {g_80} generations\n≈ {t_80:.1f} s",
    va="center",
    fontsize=14,
    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="gray",
        alpha=0.9,
    ),
)

# =========================
# Axes & formatting
# =========================
ax.set_xlabel("Planning Budget (Generation)", fontweight="bold")
ax.set_ylabel("Best Makespan (s)", fontweight="bold")

ax.set_xlim(left=0)
ax.margins(x=0)

ax.tick_params(axis="x", width=1.5)
ax.tick_params(axis="y", width=1.5)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(True, linestyle="--", alpha=0.7, zorder=0)

# =========================
# Legend (top, horizontal)
# =========================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=4,
    frameon=False,
)

# =========================
# Save
# =========================
fig.tight_layout()
fig.savefig("anytime_pareto_with_best_known_32gpu.pdf", dpi=300, bbox_inches="tight")
fig.savefig("anytime_pareto_with_best_known_32gpu.png", dpi=300, bbox_inches="tight")
plt.show()
