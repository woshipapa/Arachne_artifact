import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import pandas as pd


def load_sequence_counts_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    lengths = np.array([float(k) for k in raw_map.keys()], dtype=np.float64)
    counts = np.array([float(v) for v in raw_map.values()], dtype=np.float64)

    valid_mask = (lengths > 0) & (counts > 0)
    return lengths[valid_mask], counts[valid_mask]


def build_combined_df(t2v_datasets, llm_datasets):
    all_records = []
    for group_name, dataset_map in [("T2V", t2v_datasets), ("LLM", llm_datasets)]:
        for dataset_name, json_path in dataset_map.items():
            try:
                lengths, counts = load_sequence_counts_json(json_path)
            except FileNotFoundError:
                print(f"[Warning] File not found: {json_path}. Skipping.")
                continue
            except json.JSONDecodeError:
                print(f"[Warning] Invalid JSON: {json_path}. Skipping.")
                continue

            for length, count in zip(lengths, counts):
                all_records.append(
                    {
                        "length": float(length),
                        "count": float(count),
                        "dataset": dataset_name,
                        "group": group_name,
                    }
                )

    if not all_records:
        raise RuntimeError("No valid sequence_count JSON loaded.")

    return pd.DataFrame(all_records)


def calculate_group_mean(combined_df, group_name):
    subset = combined_df[combined_df["group"] == group_name]
    if subset.empty:
        return None
    return float(np.average(subset["length"], weights=subset["analysis_weight"]))


def apply_analysis_weights(combined_df, llm_relative_scale, reference_count):
    df = combined_df.copy()
    df["analysis_weight"] = df["count"].astype(np.float64)

    llm_totals = (
        df[df["group"] == "LLM"]
        .groupby("dataset")["count"]
        .sum()
        .to_dict()
    )
    for dataset_name, relative_scale in llm_relative_scale.items():
        dataset_total = llm_totals.get(dataset_name, 0.0)
        if dataset_total <= 0:
            continue
        target_total = float(reference_count) * float(relative_scale)
        multiplier = target_total / float(dataset_total)
        mask = (df["group"] == "LLM") & (df["dataset"] == dataset_name)
        df.loc[mask, "analysis_weight"] = df.loc[mask, "count"] * multiplier

    return df


def plot_combined_kde(combined_df, output_path, t2v_datasets, llm_datasets):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Verdana"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 6))

    manual_colors = {
        "WebVid": "#1b9e77",
        "Koala-36M": "#6A0DAD",
        "1080p": "#d95f02",
        "CommonCrawl": "#a6cee3",
        "GitHub": "#1f78b4",
    }
    datasets = sorted(combined_df["dataset"].unique())
    color_map = {name: manual_colors.get(name, "#808080") for name in datasets}

    for name in datasets:
        subset = combined_df[combined_df["dataset"] == name]
        display_name = "Lynx" if name == "1080p" else name
        sns.kdeplot(
            data=subset,
            x="length",
            weights="analysis_weight",
            ax=ax,
            log_scale=True,
            fill=True,
            alpha=0.2,
            linewidth=2.5,
            common_norm=False,
            color=color_map.get(name),
            label=display_name,
        )

    def format_ticks(value, _):
        if value >= 1000:
            return f"{value/1000:.0f}K"
        return f"{value:.0f}"

    ticks = [2**i for i in range(7, 19)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.tick_params(axis="x", rotation=45, labelsize=16)
    ax.set_xlim(left=128, right=300000)
    ax.set_xlabel("Sequence Length", fontweight="bold", fontsize=20)
    ax.set_ylabel("Probability Density", fontweight="bold", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    llm_avg = calculate_group_mean(combined_df, "LLM")
    t2v_avg = calculate_group_mean(combined_df, "T2V")
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")

    if llm_avg is not None:
        ax.axvline(x=llm_avg, color="#00008B", linestyle="--", linewidth=2.5)
        ax.text(
            llm_avg,
            ax.get_ylim()[1] * 0.65,
            f"LLM Mean\n({llm_avg:,.0f})",
            color="#00008B",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
            bbox=bbox_props,
        )

    if t2v_avg is not None:
        ax.axvline(x=t2v_avg, color="#B22222", linestyle="--", linewidth=2.5)
        ax.text(
            t2v_avg,
            ax.get_ylim()[1] * 0.45,
            f"T2V Mean\n({t2v_avg:,.0f})",
            color="#B22222",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
            bbox=bbox_props,
        )

    handles, labels = ax.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))
    t2v_display_labels = sorted(["Lynx" if n == "1080p" else n for n in t2v_datasets.keys()])
    llm_labels_sorted = sorted([n for n in llm_datasets.keys()])
    t2v_handles = [handle_map[n] for n in t2v_display_labels if n in handle_map]
    llm_handles = [handle_map[n] for n in llm_labels_sorted if n in handle_map]

    legend1 = ax.legend(
        llm_handles,
        llm_labels_sorted,
        title="LLM Datasets",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=14,
        title_fontsize=14,
    )
    plt.setp(legend1.get_title(), fontweight="bold")
    ax.add_artist(legend1)

    legend2 = ax.legend(
        t2v_handles,
        t2v_display_labels,
        title="T2V Datasets",
        loc="upper left",
        bbox_to_anchor=(0, 0.80),
        fontsize=14,
        title_fontsize=14,
    )
    plt.setp(legend2.get_title(), fontweight="bold")
    ax.add_artist(legend2)

    sns.despine(top=False, right=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_cdf(combined_df, output_path, t2v_datasets, llm_datasets):
    sns.set_theme(
        style="ticks",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Verdana"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        },
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    t2v_colors = sns.color_palette("Greens_d", n_colors=len(t2v_datasets))
    llm_colors = sns.color_palette("Blues_d", n_colors=len(llm_datasets))
    color_map = {
        **dict(zip(sorted(t2v_datasets.keys()), t2v_colors)),
        **dict(zip(sorted(llm_datasets.keys()), llm_colors)),
    }

    datasets = sorted(combined_df["dataset"].unique())
    for name in datasets:
        subset = combined_df[combined_df["dataset"] == name]
        display_name = "Lynx" if name == "1080p" else name
        sns.ecdfplot(
            data=subset,
            x="length",
            weights="analysis_weight",
            ax=ax,
            log_scale=True,
            linewidth=2.5,
            color=color_map.get(name),
            label=display_name,
        )

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("")
    ax.set_yticks([])

    def format_ticks(value, _):
        if value >= 1000:
            return f"{int(value/1000)}K"
        return str(int(value))

    ticks = [2**i for i in range(7, 19)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    max_len = combined_df["length"].max()
    ax.set_xlim(left=128, right=max_len)
    ax.set_xlabel("Sequence Lengths", fontsize=18)
    ax.set_title("CDF of Sequence Lengths", fontsize=22)

    handles, labels = ax.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))
    t2v_display_labels = sorted(["Lynx" if n == "1080p" else n for n in t2v_datasets.keys()])
    llm_labels_sorted = sorted([n for n in llm_datasets.keys()])
    t2v_handles = [handle_map[n] for n in t2v_display_labels if n in handle_map]
    llm_handles = [handle_map[n] for n in llm_labels_sorted if n in handle_map]

    legend1 = ax.legend(
        llm_handles,
        llm_labels_sorted,
        title="LLM Datasets",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=14,
        title_fontsize=14,
    )
    ax.add_artist(legend1)
    ax.legend(
        t2v_handles,
        t2v_display_labels,
        title="T2V Datasets",
        loc="upper left",
        bbox_to_anchor=(0.25, 1),
        fontsize=14,
        title_fontsize=14,
    )

    sns.despine(top=False, right=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    t2v_datasets = {
        "Koala-36M": "koala-36M_sequence_counts.json",
        "1080p": "1080p_sequence_counts.json",
    }
    llm_datasets = {
        "CommonCrawl": "CommonCrawl_sequence_counts.json",
        "GitHub": "GitHub_sequence_counts.json",
    }
    llm_relative_scale = {
        "CommonCrawl": 10.0,
        "GitHub": 2.0,
    }
    reference_dataset_name = "Koala-36M"
    output_prefix = "json_only_comparison"

    combined_df = build_combined_df(t2v_datasets, llm_datasets)
    reference_count = combined_df[combined_df["dataset"] == reference_dataset_name]["count"].sum()
    if reference_count <= 0:
        raise RuntimeError(f"Reference dataset '{reference_dataset_name}' has zero/empty counts.")

    combined_df = apply_analysis_weights(
        combined_df,
        llm_relative_scale=llm_relative_scale,
        reference_count=reference_count,
    )

    llm_mean = calculate_group_mean(combined_df, "LLM")
    t2v_mean = calculate_group_mean(combined_df, "T2V")

    print(f"LLM Mean Length: {llm_mean:.2f}" if llm_mean is not None else "LLM Mean Length: N/A")
    print(f"T2V Mean Length: {t2v_mean:.2f}" if t2v_mean is not None else "T2V Mean Length: N/A")

    plot_combined_kde(combined_df, f"{output_prefix}_kde.pdf", t2v_datasets, llm_datasets)
    plot_combined_cdf(combined_df, f"{output_prefix}_cdf.pdf", t2v_datasets, llm_datasets)
