from pathlib import Path
import json
import re
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

METRICS_DIR = Path("reports/metrics")
FNAME_RE = re.compile(r"^(?P<model>.+)_(?P<dataset>.+)_(?P<subset>.+)_(?P<split>.+)_metrics\.json$")

def load_metrics_flat(metrics_dir: Path = METRICS_DIR, regex: re.Pattern = FNAME_RE) -> pd.DataFrame:
    files = sorted(metrics_dir.glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No *_metrics.json in {metrics_dir}")

    rows = []
    for p in files:
        with open(p) as f:
            obj: Dict[str, Any] = json.load(f)

        # metadata from filename
        meta = {"model": None, "dataset": None, "subset": None, "split": None}
        m = regex.match(p.name)
        if m:
            meta.update(m.groupdict())

        # prefer metadata inside JSON if present
        for k in meta.keys():
            if obj.get(k):
                meta[k] = obj[k]

        # flatten metrics block; keep keys lowercase
        metrics = {k.lower(): v for k, v in obj.get("metrics", {}).items()}

        # also support minimal schema directly at root
        for k in ("wer", "cer"):
            if k in obj and k not in metrics:
                metrics[k] = obj[k]

        row = {**meta, **metrics, "file": str(p)}
        rows.append(row)

    df = pd.DataFrame(rows)

    # ensure numeric types for all metric-like columns
    metric_cols = [c for c in df.columns if c not in {"model", "dataset", "subset", "split", "file"}]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # stable column order
    front = ["model", "dataset", "subset", "split", "wer", "cer", "file"]
    cols = front + [c for c in df.columns if c not in front]
    df = df.reindex(columns=cols)

    return df

def compare_eval_results(metrics_dir: Path, regex: re.Pattern, ) -> pd.DataFrame:
    """Load and compare evaluation results from JSON files.

    Returns:
        A DataFrame containing the comparison of evaluation results.
    """
    df = load_metrics_flat(metrics_dir=metrics_dir, regex=regex)
    
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Order models by average WER to make the chart easier to read
    model_order = df.groupby("model")["wer"].mean().sort_values().index

    plt.figure(figsize=(12,5))
    sns.barplot(
        data=df,
        x="model", y="wer",
        hue="dataset",
        order=model_order,
        estimator="mean",
        errorbar="se"
    )
    plt.title("WER by model grouped by dataset")
    plt.ylabel("WER")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("reports/figures/wer_by_model_grouped_by_dataset.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_wer_by_model_grouped_with_error(df)

    return df

def plot_wer_by_model_grouped_with_error(df, save_path="reports/figures/wer_by_model_grouped_with_error.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # one row per model x dataset with mean wer and the provided wer_std
    agg = df.groupby(["model", "dataset"], as_index=False).agg(
        wer=("wer", "mean"),
        wer_sem=("wer_sem", "mean")  # use your precomputed std from the metrics files
    )

    model_order = agg.groupby("model")["wer"].mean().sort_values().index
    hue_order = agg["dataset"].unique()

    ax = sns.barplot(
        data=agg,
        x="model", y="wer",
        hue="dataset",
        order=model_order,
        hue_order=hue_order,
        errorbar=None  # we'll add our own error bars from wer_sem
    )
    ax.set_title("WER by model grouped by dataset")
    ax.set_ylabel("WER")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # overlay error bars per hue using the bar centers
    # seaborn groups bars into ax.containers, one container per hue level
    sem_map = agg.set_index(["model", "dataset"])["wer_sem"].to_dict()

    for container, dataset in zip(ax.containers, hue_order):
        # bars are in model_order within each container
        x = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y = [bar.get_height() for bar in container]
        yerr = [sem_map.get((m, dataset), np.nan) for m in model_order]
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# usage:




