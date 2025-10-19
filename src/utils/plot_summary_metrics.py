import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from scipy import stats
from loguru import logger

sns.set(style="whitegrid")

# =========================
# Configuration
# =========================
MODELS = [
    "roest-wav2vec2-315m-v2",
    "roest-wav2vec2-1B-v2",
    "roest-wav2vec2-2B-v2",
    "hviske-v3-conversation",
    "hviske-v2",
    "roest-whisper-large-v1",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "seamless-m4t-v2-large",
    "parakeet-tdt-0.6b-v3",
    "canary-1b-v2",
]

DATASETS = ["coral-v2", "fleurs"]

SUBSETS = {
    "coral-v2": "read_aloud",
    "fleurs": "da_dk",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
}

EVALUATION_COMBINATIONS = [
    {
        "model": m,
        "dataset_name": d,
        "dataset_subset": SUBSETS[d],
        "dataset_split": SPLITS[d],
    }
    for m in MODELS
    for d in DATASETS
]

FORMAT_DICT = {
    "co2_g": f"$CO_2$ Emissions (g)",
    "RTFx": "Inverse Real-Time Factor (RTFx)",
    "WER": "Word Error Rate (WER)",
    "CER": "Character Error Rate (CER)",
    "dataset_name": "Dataset",
    "model": "Model",
    "coral-v2": "CoRal v2",
    "fleurs": "Fleurs",
}

def build_model_handles(model_order: Sequence[str], model_palette: Dict[str, str]) -> List[Line2D]:
        return [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=model_palette[m], markeredgecolor="black",
                   markersize=8)
            for m in model_order
        ]

def build_dataset_handles(ds_order: Sequence[str], ds_markers: Dict[str, str]) -> List[Line2D]:
    return [
        Line2D([0], [0], marker=ds_markers[d], linestyle="None",
                markerfacecolor="white", markeredgecolor="black",
                markersize=8)
        for d in ds_order
    ]


def _fmt(label: str) -> str:
    return FORMAT_DICT.get(label, label)


def plot_summary_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    save_dir: Optional[str] = None,
    width: int = 8,
    height: int = 6,
    alpha: float = 0.9,
    point_size: int = 100,
    fontsize: int = 12,
    separate_by_dataset: bool = False,
    add_labels: bool = False,
    models_order: Optional[Sequence[str]] = None,
    model_legend_loc: str = "upper left",
    dataset_legend_loc: str = "lower right",
):
    """
    Scatter for summary dataframe.
    Colors = model, shapes = dataset.
    Legends are inside the plot.
    All axis/legend/facet labels go through FORMAT_DICT.
    """
    data = df.copy()

    # stable orders
    if models_order is not None:
        model_order = [m for m in models_order if m in data["model"].unique()]
    elif isinstance(data["model"].dtype, pd.CategoricalDtype):
        model_order = list(data["model"].cat.categories)
    else:
        model_order = sorted(data["model"].unique())

    if isinstance(data["dataset_name"].dtype, pd.CategoricalDtype):
        ds_order = list(data["dataset_name"].cat.categories)
    else:
        ds_order = sorted(data["dataset_name"].unique())

    # palettes and markers
    model_palette = dict(zip(model_order, sns.color_palette("tab20", n_colors=len(model_order))))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    ds_markers = dict(zip(ds_order, markers[: len(ds_order)]))

    # formatted strings
    x_label = _fmt(x)
    y_label = _fmt(y)
    model_title = _fmt("model")
    dataset_title = _fmt("dataset_name")

    if not separate_by_dataset:
        plt.figure(figsize=(width, height))
        ax = sns.scatterplot(
            data=data,
            x=x, y=y,
            hue="model",
            style="dataset_name",
            hue_order=model_order,
            style_order=ds_order,
            palette=model_palette,
            markers=ds_markers,
            s=point_size,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            legend=False,
        )

        ax.set_title(f"{y_label} vs {x_label}", fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

        # Model legend (formatted labels)
        leg_model = ax.legend(
            handles=build_model_handles(model_order=model_order, model_palette=model_palette),
            labels=[_fmt(m) for m in model_order],
            title=model_title,
            loc=model_legend_loc,
            fontsize=fontsize,
            frameon=True,
        )
        ax.add_artist(leg_model)

        # Dataset legend (formatted labels)
        leg_dataset = ax.legend(
            handles=build_dataset_handles(ds_order=ds_order, ds_markers=ds_markers),
            labels=[_fmt(d) for d in ds_order],
            title=dataset_title,
            loc=dataset_legend_loc,
            fontsize=fontsize,
            frameon=True,
        )
        ax.add_artist(leg_dataset)

        if add_labels:
            for _, r in data.iterrows():
                ax.text(r[x], r[y], _fmt(r["model"]), fontsize=fontsize, alpha=0.75)

        fig = ax.get_figure()

    else:
        g = sns.FacetGrid(
            data, col="dataset_name", col_order=ds_order,
            height=height, aspect=width / (height * max(1, len(ds_order))),
            sharex=False, sharey=True
        )
        g.map_dataframe(
            sns.scatterplot,
            x=x, y=y,
            hue="model",
            hue_order=model_order,
            palette=model_palette,
            s=point_size,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            legend=False,
        )

        # Axis labels (formatted)
        g.set_axis_labels(x_label, y_label)
        for ax in g.axes.flat:
            ax.set_xlabel(x_label, fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)

        # Facet titles: format dataset names
        for ax, ds in zip(g.axes.flat, ds_order):
            ax.set_title(_fmt(ds), fontsize=fontsize)

        g.fig.suptitle(f"{y_label} vs {x_label} by {dataset_title}", y=1.03, fontsize=fontsize)

        # Model legend on first facet, with formatted model labels
        ax0 = g.axes.flat[-1]
        leg_model = ax0.legend(
            handles=build_model_handles(model_order=model_order, model_palette=model_palette),
            labels=[_fmt(m) for m in model_order],
            title=model_title,
            loc=model_legend_loc,
            fontsize=fontsize,
            frameon=True,
        )
        ax0.add_artist(leg_model)

        if add_labels:
            for ax, ds in zip(g.axes.flat, ds_order):
                sub = data[data["dataset_name"] == ds]
                for _, r in sub.iterrows():
                    ax.text(r[x], r[y], _fmt(r["model"]), fontsize=7, alpha=0.75)

        fig = g.fig

    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{y}_vs_{x}{'_faceted' if separate_by_dataset else ''}.png"
        fig.savefig(Path(save_dir) / filename, dpi=200, bbox_inches="tight")
    plt.close()


def make_all_summary_plots(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    width: int = 10,
    height: int = 6,
    alpha: float = 0.9,
    point_size: int = 100,
    separate_by_dataset: bool = False,
    models_order: Optional[Sequence[str]] = None,
    model_legend_loc: str = "upper left",
    dataset_legend_loc: str = "lower right",
    fontsize: int = 14
) -> None:
    
    """
    Make all summary plots for given metrics.
    Args:
        df: Summary dataframe with metrics and metadata.
        save_dir: Directory to save plots. If None, plots are not saved.
        width: Width of the plots.
        height: Height of the plots.
        alpha: Alpha transparency for points.
        point_size: Size of the scatter points.
        separate_by_dataset: Whether to create faceted plots by dataset.
        models_order: Optional order of models for consistent coloring.
        model_legend_loc: Location of the model legend.
        dataset_legend_loc: Location of the dataset legend.
    """
    summary_df = df.copy()

    # WER vs CO2
    logger.info("Plotting WER vs CO2 in same facet...")
    plot_summary_scatter(
        summary_df, 
        x="co2_g", 
        y="WER", 
        model_legend_loc="upper right", 
        dataset_legend_loc="lower right", 
        fontsize=fontsize, 
        save_dir=save_dir,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        separate_by_dataset=False,
        models_order=models_order,
    )

    logger.info("Plotting WER vs CO2 faceted by dataset...")
    plot_summary_scatter(
        summary_df, 
        x="co2_g", 
        y="WER", 
        model_legend_loc="upper right", 
        dataset_legend_loc="lower right", 
        fontsize=fontsize, 
        save_dir=save_dir,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        separate_by_dataset=True,
        models_order=models_order,
    )
    # WER vs RTFx
    logger.info("Plotting WER vs RTFx...")
    plot_summary_scatter(
        summary_df, 
        x="RTFx", 
        y="WER", 
        model_legend_loc="upper center", 
        dataset_legend_loc="lower right", 
        fontsize=fontsize, 
        save_dir=save_dir,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        separate_by_dataset=separate_by_dataset,
        models_order=models_order,
    )
    