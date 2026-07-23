# utils/profiler_tools.py

import pandas as pd
from pathlib import Path
from typing import Union, Optional

def analyze_sm_throughput_from_csv(
    csv_path: Union[str, Path],
    sm_metric: str = "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    output_path: Optional[Union[str, Path]] = None,
    print_top: int = 10,
    min_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Analyse SM throughput in an NCU-exported CSV, aggregated per kernel.
    
    Args:
        csv_path (str or Path): path to the NCU CSV.
        sm_metric (str): metric name to analyse.
        output_path (Optional[str or Path]): where to save the result CSV, if given.
        print_top (int): how many aggregated rows to print.
        min_threshold (Optional[float]): if set, ignore metrics below this value.
        
    Returns:
        pd.DataFrame: per-kernel aggregated statistics.
    """
    df = pd.read_csv(csv_path)
    
    # select the metric rows
    df_sm = df[df["Metric Name"] == sm_metric].copy()
    df_sm["Metric Value"] = pd.to_numeric(df_sm["Metric Value"], errors="coerce")

    if min_threshold is not None:
        df_sm = df_sm[df_sm["Metric Value"] > min_threshold]

    overall_mean = df_sm["Metric Value"].mean()
    print(f"[{Path(csv_path).name}] Overall avg SM throughput: {overall_mean:.2f}%")

    # aggregate per kernel
    result = (
        df_sm.groupby("Kernel Name")["Metric Value"]
        .agg(["count", "mean", "max", "min", "sum"])
        .sort_values(by="mean", ascending=False)
    )
    # pd.set_option("display.max_colwidth", 100)
    if print_top > 0:
        print(f"\nTop {print_top} kernels by mean SM throughput:")
        print(result.head(print_top))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path)
        print(f"\n[+] Saved kernel summary to: {output_path}")

    return result


def compare_kernel_metrics(csv1_path, csv2_path, sm_metric="sm__throughput.avg.pct_of_peak_sustained_elapsed", print_top=20):
    def get_result(csv_path):
        df = pd.read_csv(csv_path)
        df_sm = df[df["Metric Name"] == sm_metric].copy()
        df_sm["Metric Value"] = pd.to_numeric(df_sm["Metric Value"], errors="coerce")
        df_sm = df_sm.dropna()

        result = (
            df_sm.groupby("Kernel Name")["Metric Value"]
            .agg(["count", "mean", "max", "min", "sum"])
            .sort_values(by="mean", ascending=False)
        )
        return result

    # read the two CSV analyses
    result1 = get_result(csv1_path).add_suffix("_1")
    result2 = get_result(csv2_path).add_suffix("_2")

    # merge on kernel name
    merged = result1.merge(result2, left_index=True, right_index=True, how="inner")

    # add diff columns (mean/min/max deltas)
    merged["mean_diff"] = merged["mean_2"] - merged["mean_1"]
    merged["min_diff"] = merged["min_2"] - merged["min_1"]
    merged["max_diff"] = merged["max_2"] - merged["max_1"]

    # sort to surface the top-k most changed kernels
    top_changed = merged.sort_values(by="mean_diff", ascending=False).head(print_top)

    print(f"\nTop {print_top} changed kernels (by mean SM throughput difference):")
    print(top_changed[["mean_1", "mean_2", "mean_diff", "min_diff", "max_diff"]])

    return top_changed
