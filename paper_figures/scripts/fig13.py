import numpy as np
import json
import matplotlib.pyplot as plt
from paper_plot_style import apply_paper_style

# =================================================
# Style
# =================================================
apply_paper_style(figsize=(7.1, 5.3))

# =================================================
# Data cache
# =================================================
DATA_JSON = "toy_optimality_gap_data.json"

# =================================================
# Data
# =================================================
with open(DATA_JSON, "r") as f:
    raw = json.load(f)
gap_results = {k: np.array(v, dtype=float) for k, v in raw.items()}

labels = list(gap_results.keys())
data = [gap_results[k] for k in labels]

# =================================================
# Plot (horizontal boxplot)
# =================================================
fig, ax = plt.subplots()

bp = ax.boxplot(
    data,
    vert=False,
    widths=0.55,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(linewidth=2, edgecolor="#808080"),
    medianprops=dict(linewidth=3.0, color="#6E6E6E"),
    whiskerprops=dict(linewidth=1.8, color="#808080"),
    capprops=dict(linewidth=1.8, color="#808080"),
)

# =================================================
# Color: all gray
# =================================================
for box in bp["boxes"]:
    box.set_facecolor("#B0B0B0")
    box.set_alpha(0.8)

# =================================================
# Axes formatting
# =================================================
ax.set_xlabel("Optimality Gap (%)", fontweight="bold")
ax.set_ylabel("# Cascades", fontweight="bold")

ax.set_yticks(range(1, len(labels) + 1))
ax.set_yticklabels(labels, fontweight="bold")

ax.set_xlim(0.0, 6.0)
ax.set_xticks(np.arange(0, 7, 1))

ax.grid(axis="x", linestyle="--", alpha=0.6)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")

# =================================================
# Save (NO bbox hacks)
# =================================================
fig.savefig(
    "toy_optimality_gap_vs_cascades.pdf",
    dpi=300
)
plt.show()
