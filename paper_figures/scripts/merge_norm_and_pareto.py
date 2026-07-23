import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Style Setting
# =========================
try:
    from paper_plot_style import apply_paper_style
    apply_paper_style(figsize=(14, 5))
except ImportError:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Verdana']})

# =========================
# Data Preparation
# =========================

# --- Data for Left Plot (Anytime) ---
INPUT_JSONL = "pareto_genetic_32gpus.jsonl"
MAX_GENERATION = 150
MEGATRON_MAKESPAN = 50.78
FLEXSP_MAKESPAN = 41.77
BEST_KNOWN_MAKESPAN = 33.454
SOLVER_TIME_LIMIT = 600
TARGET_RATIO = 0.90
ANNOTATION_TEXT_OFFSET = (5, 2.0)

gens = np.arange(0, MAX_GENERATION + 1)
median = 55 * np.exp(-0.05 * gens) + 33
median_time = gens * 0.5 
# ------------------------------------

# --- Data for Right Plot (Gap) ---
genetic_makespans = [17.90, 39.40, 24.91, 31.23, 23.92, 15.41, 31.19, 20.58, 14.39, 26.64]
optimal_makespans = [17.56, 38.30, 24.20, 30.30, 23.07, 15.09, 30.05, 19.76, 14.08, 25.64]

genetic = np.array(genetic_makespans)
optimal = np.array(optimal_makespans)
gap_iters = np.arange(1, len(genetic) + 1)
gap = (genetic - optimal) / optimal * 100.0


# =========================
# Plotting Combined Figure
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------------------------------------------------
# Left Plot: Anytime Curve
# -------------------------------------------------------------------
# (A) Arachne Curve
ax1.plot(gens, median, linewidth=2.5, color='#D62728', label="Arachne (Ours)", zorder=5)

# (B) Baselines
ax1.axhline(MEGATRON_MAKESPAN, linestyle=":", linewidth=2, color='gray', label="Megatron-LM", zorder=2)
ax1.axhline(FLEXSP_MAKESPAN, linestyle="--", linewidth=2, color='tab:blue', label="FlexSP", zorder=2)
ax1.axhline(BEST_KNOWN_MAKESPAN, linestyle="-.", linewidth=2, color='black', label=f"Best Known\n(Solver {SOLVER_TIME_LIMIT}s)", zorder=1)

# (C) Target Calculation & Annotation
target_makespan = BEST_KNOWN_MAKESPAN / TARGET_RATIO
try:
    idx = np.where(median <= target_makespan)[0][0]
    g_target = gens[idx]
    t_target = median_time[idx]
    val_target = median[idx]
    
    ax1.axvline(g_target, linestyle="--", linewidth=1.5, color='gray', alpha=0.8, zorder=1)
    ax1.scatter([g_target], [val_target], color='black', s=50, zorder=6)
    
    pct = int(TARGET_RATIO * 100)
    ax1.annotate(
        f"Reaches {pct}% of Best Known\n≈ {g_target} gens\n≈ {t_target:.1f} s",
        xy=(g_target, val_target), 
        xytext=(g_target + ANNOTATION_TEXT_OFFSET[0], val_target + ANNOTATION_TEXT_OFFSET[1]),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        verticalalignment='bottom'
    )
except:
    pass

# Formatting Left
ax1.set_xlabel("Planning Budget (Generation)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Best Makespan (s)", fontsize=14, fontweight="bold")
ax1.set_xlim(left=0)
ax1.margins(x=0)
ax1.grid(True, linestyle='--', alpha=0.6)

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=11, frameon=False)


# -------------------------------------------------------------------
# Right Plot: Optimality Gap
# -------------------------------------------------------------------
ax2.plot(gap_iters, gap, marker="o", linewidth=2.2, markersize=6, label="Optimality Gap", zorder=3, color='tab:green')
ax2.axhline(0.0, linestyle="--", linewidth=1.5, color="gray", zorder=1)

# Formatting Right
ax2.set_xlabel("Iteration", fontsize=14, fontweight="bold")
ax2.set_ylabel("Optimality Gap (%)", fontsize=14, fontweight="bold")
ax2.set_xlim(1, gap_iters[-1])
ax2.set_ylim(bottom=0)
ax2.margins(x=0)
ax2.grid(True, linestyle="--", alpha=0.7, zorder=0)

# Legend Right
ax2.legend(loc="upper right", fontsize=11, frameon=False)

# =========================
# Global Formatting
# =========================
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

plt.tight_layout()

plt.savefig("combined_analysis_figure.pdf", dpi=300, bbox_inches="tight")
plt.savefig("combined_analysis_figure.png", dpi=300, bbox_inches="tight")
plt.show()