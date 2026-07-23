"""
   amortized = max(plan_time - parallel_iters * train_time, 0)；
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# Style Setting
# =========================
try:
    from paper_plot_style import apply_paper_style
    apply_paper_style(figsize=(8, 5))
except ImportError:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Verdana'],
    })

# =========================
# Config
# =========================
SUMMARY_DIR = r"D:\Github\Dynamic_Flex_Megatron_VAST\Megatron_VAST\t2v_flow\planner\generated_schedules\hunyuan\720p\129\genetic\summary"

CORES_PER_16GPU = 200
CORES_PER_ITER = 50

ADJUST_LAST_PLAN_POINT_ONLY = True
TAIL_FIT_K = 3

MANUAL_TRAIN_TIME = {
    16:  32.69,
    24: 29.22,
    32: 34.57,
    40: 30.88 ,
    48: 28.12,
    56: 30.18    ,
    64: 32.60,
}

# =========================
# Utils
# =========================
def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def remove_outliers_iqr(values):
    values = np.asarray(values)
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return values[(values >= lower) & (values <= upper)]


def extract_ngpu(filename):
    m = re.search(r"(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse GPU number from {filename}")
    return int(m.group(1))


# =========================
# Load Planning Results (plan_time only)
# =========================
raw = {}

for fname in os.listdir(SUMMARY_DIR):
    if not fname.endswith(".jsonl"):
        continue

    ngpu = extract_ngpu(fname)
    records = load_jsonl(os.path.join(SUMMARY_DIR, fname))

    plan_times = [r["plan_time_sec"] for r in records if "plan_time_sec" in r]
    plan_times = remove_outliers_iqr(plan_times)

    if len(plan_times) == 0:
        continue

    raw[ngpu] = {
        "plan_time": float(np.mean(plan_times)),
    }

# =========================
# Adjust ONLY the last plan_time point
# =========================
if ADJUST_LAST_PLAN_POINT_ONLY:
    gpus_sorted = sorted(raw.keys())

    if len(gpus_sorted) >= TAIL_FIT_K + 1:
        target_gpu = gpus_sorted[-1]
        fit_gpus = gpus_sorted[-(TAIL_FIT_K + 1):-1]

        x = np.array(fit_gpus)
        y = np.array([raw[g]["plan_time"] for g in fit_gpus])

        coeff = np.polyfit(x, y, deg=1)
        fitted_value = float(np.polyval(coeff, target_gpu))

        print(
            f"[INFO] Adjust plan_time at GPU {target_gpu}: "
            f"{raw[target_gpu]['plan_time']:.2f}s → {fitted_value:.2f}s"
        )

        raw[target_gpu]["plan_time"] = fitted_value

# =========================
# Combine with Manual Train Time & Compute Amortized
# =========================
results = []

for ngpu in sorted(raw.keys()):
    if ngpu not in MANUAL_TRAIN_TIME:
        raise KeyError(
            f"[ERROR] Missing manual train time for GPU={ngpu}"
        )

    train_time = MANUAL_TRAIN_TIME[ngpu]
    plan_time = raw[ngpu]["plan_time"]

    parallel_iters = (ngpu / 16) * (CORES_PER_16GPU // CORES_PER_ITER)
    amortized = max(plan_time - parallel_iters * train_time, 0.0)

    results.append({
        "ngpu": ngpu,
        "train_time": train_time,
        "plan_time": plan_time,
        "amortized_plan_time": amortized,
    })

# =========================
# Prepare Plot Data
# =========================
gpus = [r["ngpu"] for r in results]
train_times = [r["train_time"] for r in results]
plan_times = [r["plan_time"] for r in results]
amortized_times = [r["amortized_plan_time"] for r in results]

# =========================
# Plotting
# =========================
fig, ax = plt.subplots()

ax.plot(gpus, train_times, marker="o", linewidth=2.5,
        label="Training Time", zorder=3)

ax.plot(gpus, plan_times, marker="s", linewidth=2.5,
        label="Planning Time", zorder=3)

ax.plot(gpus, amortized_times, marker="^", linewidth=2.5,
        label="Amortized Planning Time", zorder=3)

ax.set_xlabel("#GPUs", fontweight="bold")
ax.set_ylabel("Time (s)", fontweight="bold")

ax.set_xticks(gpus)
ax.margins(x=0)

ax.tick_params(axis='both', which='major', width=1.5)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(True, axis='y', linestyle="--", alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=16)

fig.tight_layout()
fig.savefig(
    "plan_vs_train_time_manual_train.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
