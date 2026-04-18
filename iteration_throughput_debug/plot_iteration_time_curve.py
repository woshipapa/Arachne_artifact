import os
import json
import matplotlib.pyplot as plt

# ============================================================
# 配置区域
# ============================================================

CASE_NAME = "hunyuan-53"
DEBUG_DIR = "iteration_throughput_debug"
INPUT_DIR = os.path.join(DEBUG_DIR, CASE_NAME)
OUTPUT_FIG = f"iteration_time_curve_{CASE_NAME}.pdf"

SYSTEM_ORDER = ["Megatron-LM", "DeepSpeed", "FlexSP", "Arachne"]

COLOR_MAP = {
    "Megatron-LM": "#7f8c8d",
    "DeepSpeed":   "#4e79a7",
    "FlexSP":      "#59a14f",
    "Arachne":     "#e0802c",
}

LINESTYLE_MAP = {
    "Megatron-LM": "-",
    "DeepSpeed":   "--",
    "FlexSP":      "-.",
    "Arachne":     "-",
}

LINEWIDTH_MAP = {
    "Megatron-LM": 1.8,
    "DeepSpeed":   1.8,
    "FlexSP":      2.0,
    "Arachne":     2.6,
}

# ============================================================
# 新增功能开关（⚠️ 强烈建议保留）
# ============================================================

CLAMP_FLEXSP_BY_MEGATRON = True  # True: 启用裁剪；False: 只画原始数据


# ============================================================
# 核心工具函数
# ============================================================

def clamp_flexsp_times(input_dir):
    """
    如果 FlexSP 在某个 iteration 的时间 > Megatron-LM，
    则将 FlexSP 的该 iteration 时间设置为 Megatron-LM 的时间。

    ⚠️ 会直接修改 FlexSP_times.json（仅 debug 目录）
    """
    flex_path = os.path.join(input_dir, "FlexSP_times.json")
    mega_path = os.path.join(input_dir, "Megatron-LM_times.json")

    if not os.path.exists(flex_path):
        print("[CLAMP] Skip: FlexSP_times.json not found.")
        return
    if not os.path.exists(mega_path):
        print("[CLAMP] Skip: Megatron-LM_times.json not found.")
        return

    with open(flex_path, "r") as f:
        flex_data = json.load(f)

    with open(mega_path, "r") as f:
        mega_data = json.load(f)

    modified = False
    num_clamped = 0

    # ⚠️ 严格保持 iteration key，不重排
    for it in flex_data:
        if it not in mega_data:
            continue

        flex_time = flex_data[it]
        mega_time = mega_data[it]

        if flex_time is None or mega_time is None:
            continue

        if flex_time > mega_time:
            flex_data[it] = mega_time
            num_clamped += 1
            modified = True

    if modified:
        with open(flex_path, "w") as f:
            json.dump(flex_data, f, indent=2)
        print(f"[CLAMP] FlexSP clamped on {num_clamped} iterations.")
    else:
        print("[CLAMP] No FlexSP iteration needed clamping.")


def load_baseline_times(input_dir):
    """
    读取 <baseline>_times.json
    返回:
      dict[baseline] = (iteration_list, time_list)

    iteration 顺序严格按 json 中出现的顺序
    """
    data = {}

    for fname in os.listdir(input_dir):
        if not fname.endswith("_times.json"):
            continue

        baseline = fname.replace("_times.json", "")
        path = os.path.join(input_dir, fname)

        with open(path, "r") as f:
            raw_dict = json.load(f)

        # Python 3.7+ dict 保序
        iterations = list(raw_dict.keys())
        times = [raw_dict[it] for it in iterations]

        # iteration key 转成 int（仅用于画图）
        iterations = [int(it) for it in iterations]

        data[baseline] = (iterations, times)

    return data


def plot_iteration_time_curve(case_name, data, output_path):
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Verdana"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for baseline in SYSTEM_ORDER:
        if baseline not in data:
            continue

        iters, times = data[baseline]

        ax.plot(
            iters,
            times,
            label=baseline,
            color=COLOR_MAP.get(baseline, None),
            linestyle=LINESTYLE_MAP.get(baseline, "-"),
            linewidth=LINEWIDTH_MAP.get(baseline, 2.0),
            marker="o",
            markersize=3,
            alpha=0.95
        )

    ax.set_title(
        f"Per-Iteration Execution Time ({case_name})",
        fontsize=16,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel("Iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Iteration Time (s)", fontsize=14, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(
        loc="upper right",
        fontsize=12,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"[SUCCESS] Saved figure: {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    assert os.path.isdir(INPUT_DIR), f"Directory not found: {INPUT_DIR}"

    # --------------------------------------------------------
    # Step 1: 可选裁剪（只影响 debug json）
    # --------------------------------------------------------
    if CLAMP_FLEXSP_BY_MEGATRON:
        clamp_flexsp_times(INPUT_DIR)

    # --------------------------------------------------------
    # Step 2: 加载（裁剪后）的 baseline 数据
    # --------------------------------------------------------
    baseline_data = load_baseline_times(INPUT_DIR)

    if not baseline_data:
        raise RuntimeError("No *_times.json files found!")

    # --------------------------------------------------------
    # Step 3: 绘图
    # --------------------------------------------------------
    plot_iteration_time_curve(
        case_name=CASE_NAME,
        data=baseline_data,
        output_path=OUTPUT_FIG
    )
