from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loguru import logger
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

MODELS = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "roest-wav2vec2-315m-v2",
    "roest-wav2vec2-1B-v2",
    "roest-wav2vec2-2B-v2",
    "hviske-v3-conversation",
    "hviske-v2",
    "roest-whisper-large-v1",
    "seamless-m4t-v2-large",
    "parakeet-tdt-0.6b-v3",
    "canary-1b-v2",
]


FORMAT_DICT = {
    "co2_g": "$CO_2$ Emissions (g)",
    "energy_kWh": "Energy Consumption (kWh)",
    "RTFx": "Inverse Real-Time Factor (RTFx)",
    "WER": "Word Error Rate (WER)",
    "CER": "Character Error Rate (CER)",
    "dataset_name": "Dataset",
    "model": "Model",
    "coral-v2": "CoRal v2",
    "fleurs": "Fleurs",
    "whisper-large-v3": "Whisper-L",
    "whisper-large-v3-turbo": "Whisper-L-Turbo",
    "roest-wav2vec2-315m-v2": "Røst-W2V2-315M",
    "roest-wav2vec2-1B-v2": "Røst-W2V2-1B",
    "roest-wav2vec2-2B-v2": "Røst-W2V2-2B",
    "hviske-v3-conversation": "Hviske-v3-Conv",
    "hviske-v2": "Hviske-v2",
    "seamless-m4t-v2-large": "Seamless-M4T-L",
    "roest-whisper-large-v1": "Røst-Whisper-L",
    "parakeet-tdt-0.6b-v3": "Parakeet-TDT",
    "canary-1b-v2": "Canary-1B",
    "parakeet-tdt-0.6b-v3_finetune": "Parakeet-TDT_FT",
    "parakeet-tdt-0.6b-v3_finetune_spec-aug": "Parakeet-TDT_FT+SA",
    "parakeet-tdt-0.6b-v3_finetune_speed-perturbations": "Parakeet-TDT_FT+SP",
    "parakeet-tdt-0.6b-v3_finetune_spec-aug_speed-perturbations": "Parakeet-TDT_FT+SA+SP",
    "canary-1b-v2_finetune": "Canary-1B_FT",
    "canary-1b-v2_finetune_spec-aug": "Canary-1B_FT+SA",
    "canary-1b-v2_finetune_speed-perturbations": "Canary-1B_FT+SP",
    "canary-1b-v2_finetune_spec-aug_speed-perturbations": "Canary-1B_FT+SA+SP",
}

def build_model_handles(model_order: Sequence[str], model_palette: Dict[str, str]) -> List[Line2D]:
        return [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=model_palette[m], markeredgecolor="black",
                   markersize=11)
            for m in model_order
        ]

def build_dataset_handles(ds_order: Sequence[str], ds_markers: Dict[str, str]) -> List[Line2D]:
    return [
        Line2D([0], [0], marker=ds_markers[d], linestyle="None",
                markerfacecolor="gray", markeredgecolor="black",
                markersize=11)
        for d in ds_order
    ]


def _fmt(label: str) -> str:
    return FORMAT_DICT.get(label, label)


def plot_summary_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    models: Optional[Sequence[str]] = None,
    save_dir: Optional[str] = None,
    width: int = 8,
    height: int = 6,
    alpha: float = 0.9,
    point_size: int = 100,
    fontsize: int = 12,
    separate_by_dataset: bool = False,
    add_labels: bool = False,
    models_order: Optional[Sequence[str]] = None
):
    """
    Scatter for summary dataframe.
    Colors = model, shapes = dataset.
    Legends are inside the plot.
    All axis/legend/facet labels go through FORMAT_DICT.
    """
    data = df.copy()

    # filter out models not in MODELS
    if models is not None:
        data = data[data["model"].isin(models)]

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
            legend=False
        )

        ax.set_title(f"{y_label} vs {x_label}", fontsize=fontsize+3)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

        # Model legend (formatted labels)
        leg_model = ax.legend(
            handles=build_model_handles(model_order=model_order, model_palette=model_palette),
            labels=[_fmt(m) for m in model_order],
            title=model_title,
            fontsize=fontsize,
            frameon=True,
        )
        ax.add_artist(leg_model)

        # Dataset legend (formatted labels)
        leg_dataset = ax.legend(
            handles=build_dataset_handles(ds_order=ds_order, ds_markers=ds_markers),
            labels=[_fmt(d) for d in ds_order],
            title=dataset_title,
            fontsize=fontsize,
            frameon=True,
        )
        ax.add_artist(leg_dataset)
        plt.setp(leg_model.get_title(), fontsize=fontsize+1) # for legend title
        plt.setp(leg_dataset.get_title(), fontsize=fontsize+1) # for legend title

        if add_labels:
            for _, r in data.iterrows():
                ax.text(r[x], r[y], _fmt(r["model"]), fontsize=fontsize, alpha=0.75)

        fig = ax.get_figure()

    else:
        # one marker per facet: first=o, second=s, rest fall back to a list
        facet_marker_pool = ["o", "s", "D", "^", "v", "P", "X"]
        facet_markers = {
            ds: facet_marker_pool[i] if i < len(facet_marker_pool) else "o"
            for i, ds in enumerate(ds_order)
        }

        g = sns.FacetGrid(
            data, col="dataset_name", col_order=ds_order,
            height=height, aspect=width / (height * max(1, len(ds_order))),
            sharex=False, sharey=True
        )

        # draw each facet with its own marker
        for ax, ds in zip(g.axes.flat, ds_order):
            sub = data[data["dataset_name"] == ds]
            sns.scatterplot(
                data=sub,
                x=x, y=y,
                hue="model",
                hue_order=model_order,
                palette=model_palette,
                s=point_size,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.4,
                legend=False,
                marker=facet_markers[ds],
                ax=ax,
            )
            # axis labels and facet title
            ax.set_xlabel(x_label, fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.set_title(_fmt(ds), fontsize=fontsize+2)

            if add_labels:
                for _, r in sub.iterrows():
                    ax.text(r[x], r[y], _fmt(r["model"]), fontsize=7, alpha=0.75)

        g.fig.suptitle(f"{y_label} vs {x_label} by {dataset_title}", y=1.03, fontsize=fontsize+3)

        # model legend on last facet
        ax0 = g.axes.flat[-1]
        leg_model = ax0.legend(
            handles=build_model_handles(model_order=model_order, model_palette=model_palette),
            labels=[_fmt(m) for m in model_order],
            title=model_title,
            fontsize=fontsize,
            frameon=True,
        )
        ax0.add_artist(leg_model)

        fig = g.fig
        plt.setp(leg_model.get_title(), fontsize=fontsize+1) # for legend title

    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{y}_vs_{x}{'_faceted' if separate_by_dataset else ''}.png"
        fig.savefig(Path(save_dir) / filename, dpi=200, bbox_inches="tight")
    plt.close()


def make_all_summary_plots(
    df: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    save_dir: Optional[str] = None,
    width: int = 12,
    height: int = 7,
    alpha: float = 0.9,
    point_size: int = 100,
    separate_by_dataset: bool = False,
    models_order: Optional[Sequence[str]] = None,
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
    """
    summary_df = df.copy()

    # WER vs CO2
    logger.info("Plotting WER vs energy in same facet...")
    plot_summary_scatter(
        summary_df, 
        models=models,
        x="energy_kWh", 
        y="WER",
        fontsize=fontsize, 
        save_dir=save_dir,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        separate_by_dataset=False,
        models_order=models_order,
    )

    logger.info("Plotting WER vs energy faceted by dataset...")
    plot_summary_scatter(
        summary_df, 
        models=models,
        x="energy_kWh", 
        y="WER",
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
        models=models,
        x="RTFx", 
        y="WER",
        fontsize=fontsize, 
        save_dir=save_dir,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        separate_by_dataset=separate_by_dataset,
        models_order=models_order,
    )
    