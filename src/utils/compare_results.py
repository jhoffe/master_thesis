from pathlib import Path
import json
import re
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    plot_wer_by_model_grouped_with_error(df)
    plot_cer_by_model_grouped_with_error(df)
    plot_semantic_distance_by_model_grouped_with_error(df)
    plot_wer_vs_rtfx(df)
    plot_wer_vs_co2(df, dataset="fleurs", log_x=False)
    plot_wer_vs_co2(df, dataset="coral-v2", log_x=False)
    plot_wer_vs_co2_both(df, log_x=True)
    return df


def _text_color_for_bar(bar):
    r, g, b, _ = bar.get_facecolor()
    # perceived luminance
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if L > 0.6 else "white"


def plot_wer_by_model_grouped_with_error(df, save_path="reports/figures/comparison/wer_by_model_grouped_with_error.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # one row per model x dataset with mean wer and the provided wer_sem
    if "rtfx" in df.columns:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            wer=("wer", "mean"),
            wer_sem=("wer_sem", "mean"),
            rtfx=("rtfx", "mean"),
        )
    else:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            wer=("wer", "mean"),
            wer_sem=("wer_sem", "mean"),
        )

    # --- fixed, deterministic orderings ---
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    fixed_dataset_order = ["fleurs", "coral-v2"]

    present_models = agg["model"].unique()
    present_datasets = agg["dataset"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]
    dataset_order = [d for d in fixed_dataset_order if d in present_datasets]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=dataset_order, ordered=True)
    agg = agg.sort_values(["model", "dataset"])

    hue_order = dataset_order

    fig, ax = plt.subplots(figsize=(12, 5))
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
    # keep your original rotation line
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # overlay error bars per hue using the bar centers
    sem_map = agg.set_index(["model", "dataset"])["wer_sem"].to_dict()
    for container, dataset in zip(ax.containers, hue_order):
        x = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y = [bar.get_height() for bar in container]
        yerr = [sem_map.get((m, dataset), np.nan) for m in model_order]
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

        
    if "rtfx" in agg.columns and agg["rtfx"].notna().any():
        rtfx_map = agg.set_index(["model", "dataset"])["rtfx"].to_dict()

        for container, dataset in zip(ax.containers, hue_order):
            for bar, m in zip(container, model_order):
                val = rtfx_map.get((m, dataset))
                if pd.isna(val):
                    continue
                h = bar.get_height()
                cx = bar.get_x() + bar.get_width() / 2
                # place near top inside the bar; fallback to middle for tiny bars
                if h > 0:
                    y = h * 0.85 if h > 0.08 else h * 0.5
                else:
                    # negative or zero heights: place just below the top edge
                    y = h * 0.15

                ax.text(
                    cx, y, f"x{val:.1f}",
                    ha="center", va="center",
                    fontsize=8,
                    color=_text_color_for_bar(bar),
                    clip_on=True,  # keep text within axes
                )
    if "rtfx" in agg.columns and agg["rtfx"].notna().any():
        ax.text(1.0, 1.02, "Labels inside bars show RTFx",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    
    # add grid
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cer_by_model_grouped_with_error(df, save_path="reports/figures/comparison/cer_by_model_grouped_with_error.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # one row per model x dataset with mean wer and the provided wer_sem
    if "rtfx" in df.columns:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            cer=("cer", "mean"),
            cer_sem=("cer_sem", "mean"),
            rtfx=("rtfx", "mean"),
        )
    else:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            cer=("cer", "mean"),
            cer_sem=("cer_sem", "mean"),
        )

    # --- fixed, deterministic orderings ---
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    fixed_dataset_order = ["fleurs", "coral-v2"]

    present_models = agg["model"].unique()
    present_datasets = agg["dataset"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]
    dataset_order = [d for d in fixed_dataset_order if d in present_datasets]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=dataset_order, ordered=True)
    agg = agg.sort_values(["model", "dataset"])

    hue_order = dataset_order

    fig, ax = plt.subplots(figsize=(12, 5))
    ax = sns.barplot(
        data=agg,
        x="model", y="cer",
        hue="dataset",
        order=model_order,
        hue_order=hue_order,
        errorbar=None  # we'll add our own error bars from cer_sem
    )
    ax.set_title("CER by model grouped by dataset")
    ax.set_ylabel("CER")
    ax.set_xlabel("")
    # keep your original rotation line
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # overlay error bars per hue using the bar centers
    sem_map = agg.set_index(["model", "dataset"])["cer_sem"].to_dict()
    for container, dataset in zip(ax.containers, hue_order):
        x = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y = [bar.get_height() for bar in container]
        yerr = [sem_map.get((m, dataset), np.nan) for m in model_order]
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

        
    if "rtfx" in agg.columns and agg["rtfx"].notna().any():
        rtfx_map = agg.set_index(["model", "dataset"])["rtfx"].to_dict()

        for container, dataset in zip(ax.containers, hue_order):
            for bar, m in zip(container, model_order):
                val = rtfx_map.get((m, dataset))
                if pd.isna(val):
                    continue
                h = bar.get_height()
                cx = bar.get_x() + bar.get_width() / 2
                # place near top inside the bar; fallback to middle for tiny bars
                if h > 0:
                    y = h * 0.85 if h > 0.08 else h * 0.5
                else:
                    # negative or zero heights: place just below the top edge
                    y = h * 0.15

                ax.text(
                    cx, y, f"x{val:.1f}",
                    ha="center", va="center",
                    fontsize=8,
                    color=_text_color_for_bar(bar),
                    clip_on=True,  # keep text within axes
                )
    if "rtfx" in agg.columns and agg["rtfx"].notna().any():
        ax.text(1.0, 1.02, "Labels inside bars show RTFx",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    
    # add grid
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_distance_by_model_grouped_with_error(
    df,
    save_path="reports/figures/comparison/semantic_distance_by_model_grouped_with_error.png",
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # one row per model x dataset with mean semantic distance and its sem
    if "rtfx" in df.columns:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            semdist=("avg_semantic_distance", "mean"),
            semdist_sem=("semantic_distance_sem", "mean"),
            rtfx=("rtfx", "mean"),
        )
    else:
        agg = df.groupby(["model", "dataset"], as_index=False).agg(
            semdist=("avg_semantic_distance", "mean"),
            semdist_sem=("semantic_distance_sem", "mean"),
        )

    # fixed, deterministic orderings
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    fixed_dataset_order = ["fleurs", "coral-v2"]

    present_models = agg["model"].unique()
    present_datasets = agg["dataset"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]
    dataset_order = [d for d in fixed_dataset_order if d in present_datasets]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=dataset_order, ordered=True)
    agg = agg.sort_values(["model", "dataset"])

    hue_order = dataset_order

    fig, ax = plt.subplots(figsize=(12, 5))
    ax = sns.barplot(
        data=agg,
        x="model", y="semdist",
        hue="dataset",
        order=model_order,
        hue_order=hue_order,
        errorbar=None,  # we will add our own error bars from semdist_sem
    )
    ax.set_title("Average semantic distance by model grouped by dataset")
    ax.set_ylabel("Semantic distance (lower is better)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # overlay error bars per hue using the bar centers
    sem_map = agg.set_index(["model", "dataset"])["semdist_sem"].to_dict()
    for container, dataset in zip(ax.containers, hue_order):
        x = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y = [bar.get_height() for bar in container]
        yerr = [sem_map.get((m, dataset), np.nan) for m in model_order]
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    # optional RTFx labels inside bars
    if "rtfx" in agg.columns and agg["rtfx"].notna().any():
        rtfx_map = agg.set_index(["model", "dataset"])["rtfx"].to_dict()

        # helper for readable text color on each bar
        def _text_color_for_bar(bar):
            fc = bar.get_facecolor()
            # luminance
            lum = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2]
            return "black" if lum > 0.6 else "white"

        for container, dataset in zip(ax.containers, hue_order):
            for bar, m in zip(container, model_order):
                val = rtfx_map.get((m, dataset))
                if pd.isna(val):
                    continue
                h = bar.get_height()
                cx = bar.get_x() + bar.get_width() / 2
                y = h * 0.85 if h > 0 else h * 0.15
                ax.text(
                    cx, y, f"x{val:.1f}",
                    ha="center", va="center",
                    fontsize=8,
                    color=_text_color_for_bar(bar),
                    clip_on=True,
                )
        ax.text(
            1.0, 1.02, "Labels inside bars show RTFx",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
        )

    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_wer_vs_rtfx(
    df,
    save_path="reports/figures/comparison/wer_vs_rtfx.png",
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # --- aggregate to one row per model x dataset ---
    agg = (df.groupby(["model", "dataset"], as_index=False)
             .agg(wer=("wer", "mean"), rtfx=("rtfx", "mean")))
    agg = agg.dropna(subset=["wer", "rtfx"])

    # --- fixed, deterministic orderings ---
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    fixed_dataset_order = ["fleurs", "coral-v2"]

    present_models = agg["model"].unique()
    present_datasets = agg["dataset"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]
    dataset_order = [d for d in fixed_dataset_order if d in present_datasets]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=dataset_order, ordered=True)
    agg = agg.sort_values(["model", "dataset"])

    # --- visual encodings: colors by model, markers by dataset ---
    # extend palette if needed
    base = sns.color_palette("tab20")
    if len(model_order) > len(base):
        # repeat without changing order to avoid color shift
        times = int(np.ceil(len(model_order) / len(base)))
        palette_list = (base * times)[:len(model_order)]
    else:
        palette_list = base[:len(model_order)]
    palette = {m: c for m, c in zip(model_order, palette_list)}

    marker_map = {"fleurs": "o", "coral-v2": "s"}
    markers = [marker_map[d] for d in dataset_order]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")  # constrained to fit outside legend

    sns.scatterplot(
        data=agg,
        x="rtfx",
        y="wer",
        hue="model",
        style="dataset",
        hue_order=model_order,
        style_order=dataset_order,
        palette=palette,
        markers=markers,
        s=90,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        legend=False,  # we build our own
    )

    # axes formatting
    ax.set_title("WER vs RTFx")
    ax.set_xlabel("RTFx (higher is better)")
    ax.set_ylabel("WER")
    if len(agg):
        ax.set_ylim(0, float(np.nanmax(agg["wer"]) * 1.1))
    ax.set_xscale("log")  # comment out if you prefer linear
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # --- legends ---
    # model legend (outside, right)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=palette[m], markeredgecolor="black", markersize=7)
        for m in model_order
    ]
    model_legend = ax.legend(
        model_handles, model_order,
        title="Model",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(model_legend)

    # dataset legend (inside, unobtrusive)
    dataset_handles = [
        Line2D([0], [0], marker=marker_map[d], linestyle="None",
               markerfacecolor="gray", markeredgecolor="black", markersize=7)
        for d in dataset_order
    ]
    ax.legend(
        dataset_handles, dataset_order,
        title="Dataset",
        loc="upper left",
        frameon=True,
    )

    # add grid
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.7)

    # save (no tight_layout with constrained layout)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_wer_vs_co2(
    df,
    dataset,  # "fleurs" or "coral-v2"
    log_x=False,  # set True if CO2 spans orders of magnitude
):
    save_path = f"reports/figures/comparison/wer_vs_co2_{dataset}.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- filter to one dataset ----
    if dataset not in df["dataset"].unique():
        raise ValueError(f"dataset '{dataset}' not found in df['dataset']")
    data = df[df["dataset"] == dataset].copy()

    # ---- aggregate to one row per model ----
    agg = (
        data.groupby(["model"], as_index=False)
            .agg(wer=("wer", "mean"), co2=("co2_g", "mean"))
    )
    agg = agg.dropna(subset=["wer", "co2"])

    # ---- fixed, deterministic ordering for models ----
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    present_models = agg["model"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg = agg.sort_values(["model"])

    # ---- visual encodings: colors by model ----
    base = sns.color_palette("tab20")
    if len(model_order) > len(base):
        times = int(np.ceil(len(model_order) / len(base)))
        palette_list = (base * times)[:len(model_order)]
    else:
        palette_list = base[:len(model_order)]
    palette = {m: c for m, c in zip(model_order, palette_list)}

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    sns.scatterplot(
        data=agg,
        x="co2",
        y="wer",
        hue="model",
        hue_order=model_order,
        palette=palette,
        s=90,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        legend=False,  # we’ll build our own below
    )

    dataset_title = "Fleurs" if dataset == "fleurs" else "CoRal v2" if dataset == "coral-v2" else dataset

    # axes formatting
    ax.set_title(f"WER vs $CO_2$ on {dataset_title}")
    ax.set_xlabel("$CO_2$ grams (lower is better)")
    ax.set_ylabel("WER")
    if len(agg):
        ax.set_ylim(0, float(np.nanmax(agg["wer"]) * 1.1))
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # ---- model legend (inside) ----
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=palette[m], markeredgecolor="black", markersize=7)
        for m in model_order
    ]
    ax.legend(
        model_handles, model_order,
        title="Model",
        frameon=True,
        fontsize=10,
        title_fontsize=12,
    )

    # save
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_wer_vs_co2_both(
    df,
    save_path="reports/figures/comparison/wer_vs_co2_both.png",
    log_x=False,
):

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # aggregate to one row per model and dataset
    agg = (
        df.groupby(["model", "dataset"], as_index=False)
          .agg(wer=("wer", "mean"), co2=("co2_g", "mean"))
          .dropna(subset=["wer", "co2"])
    )

    # fixed orders
    fixed_model_order = [
        "canary-1b-v2",
        "parakeet-tdt-0.6b-v3",
        "seamless-m4t-v2-large",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "hviske-v3-conversation",
        "hviske-v2",
        "roest-whisper-large-v1",
        "roest-wav2vec2-2B-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-315m-v2",
    ]
    fixed_dataset_order = ["fleurs", "coral-v2"]

    present_models = agg["model"].unique()
    present_datasets = agg["dataset"].unique()
    model_order = [m for m in fixed_model_order if m in present_models]
    dataset_order = [d for d in fixed_dataset_order if d in present_datasets]

    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=dataset_order, ordered=True)
    agg = agg.sort_values(["model", "dataset"])

    # colors by model
    base = sns.color_palette("tab20")
    if len(model_order) > len(base):
        times = int(np.ceil(len(model_order) / len(base)))
        palette_list = (base * times)[:len(model_order)]
    else:
        palette_list = base[:len(model_order)]
    palette = {m: c for m, c in zip(model_order, palette_list)}

    # markers by dataset
    marker_map = {"fleurs": "o", "coral-v2": "s"}
    markers = [marker_map[d] for d in dataset_order]

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    sns.scatterplot(
        data=agg,
        x="co2",               # CO2 on x
        y="wer",               # WER on y
        hue="model",
        style="dataset",
        hue_order=model_order,
        style_order=dataset_order,
        palette=palette,
        markers=markers,
        s=90,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        legend=False,          # build two compact legends inside
    )

    # axes
    ax.set_title("WER vs $CO_2$")
    ax.set_xlabel("$CO_2$ grams (lower is better)")
    ax.set_ylabel("WER")
    if len(agg):
        ax.set_ylim(0, float(np.nanmax(agg["wer"]) * 1.1))
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # legends inside
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=palette[m], markeredgecolor="black", markersize=7)
        for m in model_order
    ]
    model_leg = ax.legend(
        model_handles, model_order,
        title="Model",
        #loc="upper right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(model_leg)

    dataset_handles = [
        Line2D([0], [0], marker=marker_map[d], linestyle="None",
               markerfacecolor="gray", markeredgecolor="black", markersize=7)
        for d in dataset_order
    ]
    ax.legend(
        dataset_handles, dataset_order,
        title="Dataset",
        loc="lower right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
