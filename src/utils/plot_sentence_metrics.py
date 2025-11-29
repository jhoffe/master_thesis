from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np

from utils.ignore_warnings import ignore_warnings

sns.set(style="whitegrid")

ignore_warnings()


# =========================
# Formatting utilities
# =========================
FORMAT_DICT = {
    "co2_g": "$CO_2$ Emissions (g)",
    "RTFx": "Inverse Real-Time Factor (RTFx)",
    "WER": "Word Error Rate (WER)",
    "CER": "Character Error Rate (CER)",
    "semantic_distance": "Semantic Distance",
    "dataset_name": "Dataset",
    "model": "Model",
    "coral-v2": "CoRal-v2",
    "fleurs": "Fleurs",
    "whisper-large-v3": "Whisper",
    "whisper-large-v3-turbo": "Whisper-Turbo",
    "roest-wav2vec2-315m-v2": "Røst-W2V2-315M",
    "roest-wav2vec2-1B-v2": "Røst-W2V2-1B",
    "roest-wav2vec2-2B-v2": "Røst-W2V2-2B",
    "hviske-v3-conversation": "Hviske-v3-Conv",
    "hviske-v2": "Hviske-v2",
    "seamless-m4t-v2-large": "SeamlessM4T",
    "roest-whisper-large-v1": "Røst-Whisper",
    "parakeet-tdt-0.6b-v3": "Parakeet-TDT",
    "canary-1b-v2": "Canary-1B",
    "parakeet_finetune": "Parakeet-TDT-FT",
    "parakeet_finetune_pitch-shift": "Parakeet-TDT-FT+PS",
    "parakeet_finetune_spec-aug": "Parakeet-TDT-FT+SA",
    "parakeet_finetune_speed-perturbations": "Parakeet-TDT-FT+SP",
    "parakeet_finetune_spec-aug_pitch-shift": "Parakeet-TDT-FT+SA+PS",
    "parakeet_finetune_spec-aug_speed-perturbations": "Parakeet-TDT-FT+SA+SP",
    "parakeet_finetune_speed-perturbations_pitch-shift": "Parakeet-TDT-FT+SP+PS",
    "parakeet_finetune_spec-aug_speed-perturbations_pitch-shift": "Parakeet-TDT-FT+SA+SP+PS",
    "canary_finetune": "Canary-1B-FT",
    "canary_finetune_pitch-shift": "Canary-1B-FT+PS",
    "canary_finetune_spec-aug": "Canary-1B-FT+SA",
    "canary_finetune_speed-perturbations": "Canary-1B-FT+SP",
    "canary_finetune_spec-aug_pitch-shift": "Canary-1B-FT+SA+PS",
    "canary_finetune_spec-aug_speed-perturbations": "Canary-1B-FT+SA+SP",
    "canary_finetune_speed-perturbations_pitch-shift": "Canary-1B-FT+SP+PS",
    "canary_finetune_spec-aug_speed-perturbations_pitch-shift": "Canary-1B-FT+SA+SP+PS",
}


def _fmt(label: str) -> str:
    return FORMAT_DICT.get(label, label)


def _format_xtick_labels(ax, rotation: int = 60, ha: str = "right") -> None:
    """Format current x-tick labels via FORMAT_DICT and apply rotation/alignment."""
    ticks = ax.get_xticklabels()
    formatted = []
    for t in ticks:
        text = t.get_text()
        formatted.append(_fmt(text))
    ax.set_xticklabels(formatted, rotation=rotation, ha=ha)


# =========================
# Plot functions
# =========================
def plot_bar_metric(
    df: pd.DataFrame,
    metric: str = "WER",
    ci_level: int = 95,
    save_dir: Optional[str] = None,
    width: int = 10,
    height: int = 6,
    separate_by_dataset: bool = False,
    capsize: float = 0.2,
    fontsize: int = 12,
):
    """
    Bar plot of mean metric per model with SEM-based symmetric intervals.
    If separate_by_dataset=True, each facet uses a single dataset color.
    Colors come from Seaborn's muted 'deep' palette to match your boxplots.
    All labels/titles use FORMAT_DICT.
    """
    data = df.copy()
    z = 1.96 if ci_level == 95 else stats.norm.ppf(0.5 + ci_level / 200)

    ds_categories = data["dataset_name"].cat.categories.tolist()
    model_order = data["model"].cat.categories.tolist()

    # muted tones like your boxplots
    ds_palette = dict(zip(ds_categories, sns.color_palette("deep", n_colors=len(ds_categories))))

    x_lab = _fmt("model")
    y_lab = _fmt(metric)

    if separate_by_dataset:
        g = sns.catplot(
            data=data,
            x="model",
            y=metric,
            col="dataset_name",
            kind="bar",
            errorbar=("se", z),
            capsize=capsize,
            height=height,
            aspect=width / height / max(1, len(ds_categories)),
            order=model_order,
            saturation=1,
        )
        # axis + tick labels
        g.set_axis_labels(x_lab, y_lab)
        for ax in g.axes.flat:
            ax.set_xlabel(x_lab, fontsize=fontsize)
            ax.set_ylabel(y_lab, fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            _format_xtick_labels(ax, rotation=60, ha="right")

        # facet titles (format dataset names)
        for ax, ds in zip(g.axes.flat, ds_categories):
            ax.set_title(_fmt(ds), fontsize=fontsize + 2)

        g.fig.suptitle(
            f"{y_lab} by {_fmt('model')} per {_fmt('dataset_name')}", y=1.02, fontsize=fontsize + 3
        )

        # single color per facet
        for ax, ds in zip(g.axes.flat, ds_categories):
            color = ds_palette[ds]
            for bar in ax.patches:
                bar.set_facecolor(color)

        fig = g.fig

    else:
        plt.figure(figsize=(width, height))
        ax = sns.barplot(
            data=data,
            x="model",
            y=metric,
            hue="dataset_name",
            errorbar=("se", z),
            capsize=capsize,
            palette=ds_palette,
            order=model_order,
            hue_order=ds_categories,
            saturation=1,
        )

        # labels/titles
        ax.set_xlabel(x_lab, fontsize=fontsize)
        ax.set_ylabel(y_lab, fontsize=fontsize)
        ax.set_title(
            f"{y_lab} by {_fmt('model')} and {_fmt('dataset_name')}", fontsize=fontsize + 3
        )
        ax.tick_params(axis="both", labelsize=fontsize)
        _format_xtick_labels(ax, rotation=60, ha="right")

        # clean legend with formatted dataset names + title
        if ax.legend_ is not None:
            handles, _labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
            ax.legend(
                handles,
                [_fmt(d) for d in ds_categories],
                title=_fmt("dataset_name"),
                loc="best",
                fontsize=fontsize,
                frameon=True,
            )

        fig = ax.get_figure()

    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{metric}_bar_by_model{'_faceted' if separate_by_dataset else ''}.png"
        fig.savefig(Path(save_dir) / fname, dpi=200, bbox_inches="tight")
    plt.close()


def _bootstrap_mean_ci(x, level=95, B=5000, rng=None):
    x = np.asarray(x)
    n = x.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    boot = np.empty(B, dtype=float)
    for b in range(B):
        sample = rng.choice(x, size=n, replace=True)
        boot[b] = sample.mean()
    alpha = 1 - level / 100
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return x.mean(), low, high


def plot_bar_metric(
    df: pd.DataFrame,
    metric: str = "WER",
    ci_level: int = 95,
    B: int = 5000,
    save_dir: Optional[str] = None,
    width: int = 10,
    height: int = 6,
    separate_by_dataset: bool = False,
    fontsize: int = 12,
    random_state: Optional[int] = None,
):
    rng = np.random.default_rng(random_state)

    data = df.copy()
    ds_categories = data["dataset_name"].cat.categories.tolist()
    model_order = data["model"].cat.categories.tolist()

    palette = dict(zip(ds_categories, sns.color_palette("deep", n_colors=len(ds_categories))))

    if separate_by_dataset:
        group_cols = ["dataset_name", "model"]
    else:
        group_cols = ["model", "dataset_name"]

    summary = (
        data.groupby(group_cols, observed=True)
        .apply(lambda g: pd.Series(
            _bootstrap_mean_ci(g[metric].to_numpy(), level=ci_level, B=B, rng=rng),
            index=["mean", "ci_low", "ci_high"])
        )
        .reset_index()
    )

    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary["dataset_name"] = pd.Categorical(summary["dataset_name"], categories=ds_categories, ordered=True)

    summary = summary.sort_values(group_cols)

    if separate_by_dataset:
        g = sns.catplot(
            data=summary,
            x="model",
            y="mean",
            col="dataset_name",
            kind="bar",
            errorbar=None,
            capsize=0.0,
            height=height,
            aspect=width / height / max(1, len(ds_categories)),
            order=model_order,
            saturation=1,
        )

        for ax, ds in zip(g.axes.flat, ds_categories):
            subset = summary[summary["dataset_name"] == ds]

            for i, model in enumerate(model_order):
                bar = ax.patches[i]

                row = subset[subset["model"] == model].iloc[0]

                mean = row["mean"]
                low = row["ci_low"]
                high = row["ci_high"]

                x_center = bar.get_x() + bar.get_width() / 2

                ax.errorbar(
                    x_center,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    capthick=1.2,
                    linewidth=1.2,
                )

            color = palette[ds]
            for bar in ax.patches:
                bar.set_facecolor(color)

            ax.set_xlabel(_fmt("model"), fontsize=fontsize)
            ax.set_ylabel(_fmt(metric), fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            _format_xtick_labels(ax, rotation=60, ha="right")
            ax.set_title(_fmt(ds), fontsize=fontsize + 2)

        g.fig.suptitle(
            f"{_fmt(metric)} by {_fmt('model')} per {_fmt('dataset_name')}",
            y=1.02,
            fontsize=fontsize + 3,
        )

        fig = g.fig

    else:
        plt.figure(figsize=(width, height))
        ax = sns.barplot(
            data=summary,
            x="model",
            y="mean",
            hue="dataset_name",
            errorbar=None,
            palette=palette,
            order=model_order,
            hue_order=ds_categories,
            saturation=1,
        )

        # FIX: use containers instead of flat indexing
        for ds_index, ds in enumerate(ds_categories):
            container = ax.containers[ds_index]
            subset = summary[summary["dataset_name"] == ds]

            for bar, (_, row) in zip(container, subset.iterrows()):
                mean = row["mean"]
                low = row["ci_low"]
                high = row["ci_high"]

                x_center = bar.get_x() + bar.get_width() / 2

                ax.errorbar(
                    x_center,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    capthick=1.2,
                    linewidth=1.2,
                )

        ax.set_xlabel(_fmt("model"), fontsize=fontsize)
        ax.set_ylabel(_fmt(metric), fontsize=fontsize)
        ax.set_title(
            f"{_fmt(metric)} by {_fmt('model')} and {_fmt('dataset_name')}",
            fontsize=fontsize + 3,
        )
        ax.tick_params(axis="both", labelsize=fontsize)
        _format_xtick_labels(ax, rotation=60, ha="right")

        if ax.legend_ is not None:
            handles, _labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
            ax.legend(
                handles,
                [_fmt(d) for d in ds_categories],
                title=_fmt("dataset_name"),
                loc="best",
                fontsize=fontsize,
                frameon=True,
            )

        fig = ax.get_figure()

    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{metric}_bar_by_model{'_faceted' if separate_by_dataset else ''}.png"
        fig.savefig(Path(save_dir) / fname, dpi=200, bbox_inches="tight")

    plt.close()


def plot_box_metric(
    df: pd.DataFrame,
    metric: str = "WER",
    save_dir: Optional[str] = None,
    width: int = 15,
    height: int = 8,
    showfliers: bool = False,
    separate_by_dataset: bool = False,
    fontsize: int = 12,  # NEW
):
    """
    Box plot of per-sample metric distribution per model.

    - separate_by_dataset=False: grouped boxplot with hue=dataset_name (two colors).
    - separate_by_dataset=True: manual faceting (one subplot per dataset),
      each subplot uses ONE fixed color for that dataset.
    All labels/titles use FORMAT_DICT.
    """
    data = df.copy()
    model_order = data["model"].cat.categories.tolist()
    ds_categories = data["dataset_name"].cat.categories.tolist()

    ds_palette = dict(zip(ds_categories, sns.color_palette("deep", n_colors=len(ds_categories))))

    x_lab = _fmt("model")
    y_lab = _fmt(metric)

    if not separate_by_dataset:
        plt.figure(figsize=(width, height))
        ax = sns.boxplot(
            data=data,
            x="model",
            y=metric,
            hue="dataset_name",
            order=model_order,
            hue_order=ds_categories,
            palette=ds_palette,
            showfliers=showfliers,
            saturation=1,
        )
        ax.set_xlabel(x_lab, fontsize=fontsize)
        ax.set_ylabel(y_lab, fontsize=fontsize)
        ax.set_title(
            f"{y_lab} by {_fmt('model')} and {_fmt('dataset_name')}", fontsize=fontsize + 3
        )
        ax.tick_params(axis="both", labelsize=fontsize)
        _format_xtick_labels(ax, rotation=60, ha="right")

        # format legend labels + title
        if ax.legend_ is not None:
            handles, _labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
            ax.legend(
                handles,
                [_fmt(d) for d in ds_categories],
                title=_fmt("dataset_name"),
                loc="best",
                fontsize=fontsize,
                frameon=True,
            )

        fig = ax.get_figure()

    else:
        # manual faceting: one subplot per dataset, single color per dataset
        n = len(ds_categories)
        fig, axes = plt.subplots(
            1, n, figsize=(width, height), sharey=True, constrained_layout=True
        )
        if n == 1:
            axes = [axes]

        for ax, ds in zip(axes, ds_categories):
            sub = data[data["dataset_name"] == ds]
            sns.boxplot(
                data=sub,
                x="model",
                y=metric,
                order=model_order,
                showfliers=showfliers,
                color=ds_palette[ds],
                ax=ax,
                saturation=1,
            )
            # outlines/medians readable
            for patch in ax.artists:
                patch.set_edgecolor("black")
            for line in ax.lines:
                line.set_color("black")

            ax.set_title(_fmt(ds), fontsize=fontsize + 2)  # formatted facet title
            ax.set_xlabel(x_lab, fontsize=fontsize)
            ax.set_ylabel(y_lab, fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            _format_xtick_labels(ax, rotation=60, ha="right")

        fig.suptitle(
            f"{y_lab} by {_fmt('model')} per {_fmt('dataset_name')}", y=1.02, fontsize=fontsize + 3
        )

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{metric}_box_by_model{'_faceted' if separate_by_dataset else ''}.png"
        if showfliers:
            fname = fname.replace(".png", "_with_outliers.png")
        fig.savefig(Path(save_dir) / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(
    data: pd.DataFrame,
    save_dir: Optional[str] = "figures_seaborn",
    ci: str = 95,
    width: int = 12,
    height: int = 6,
) -> None:
    """
    Convenience wrapper that filters to your grid and emits all standard plots.
    """

    plot_bar_metric(data, "CER", ci_level=ci, save_dir=save_dir, width=width, height=height)
    plot_bar_metric(
        data,
        "CER",
        ci_level=ci,
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
    )
    plot_bar_metric(data, "WER", ci_level=ci, save_dir=save_dir, width=width, height=height)
    plot_bar_metric(
        data,
        "WER",
        ci_level=ci,
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
    )
    plot_bar_metric(
        data, "semantic_distance", ci_level=ci, save_dir=save_dir, width=width, height=height
    )
    plot_bar_metric(
        data,
        "semantic_distance",
        ci_level=ci,
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
    )

    plot_box_metric(data, "CER", save_dir=save_dir, width=width, height=height)
    plot_box_metric(
        data, "CER", separate_by_dataset=True, save_dir=save_dir, width=width, height=height
    )
    plot_box_metric(data, "WER", save_dir=save_dir, width=width, height=height)
    plot_box_metric(
        data, "WER", separate_by_dataset=True, save_dir=save_dir, width=width, height=height
    )
    plot_box_metric(data, "semantic_distance", save_dir=save_dir, width=width, height=height)
    plot_box_metric(
        data,
        "semantic_distance",
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
    )

    plot_box_metric(data, "CER", save_dir=save_dir, width=width, height=height, showfliers=True)
    plot_box_metric(
        data,
        "CER",
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
        showfliers=True,
    )
    plot_box_metric(data, "WER", save_dir=save_dir, width=width, height=height, showfliers=True)
    plot_box_metric(
        data,
        "WER",
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
        showfliers=True,
    )
    plot_box_metric(
        data, "semantic_distance", save_dir=save_dir, width=width, height=height, showfliers=True
    )
    plot_box_metric(
        data,
        "semantic_distance",
        separate_by_dataset=True,
        save_dir=save_dir,
        width=width,
        height=height,
        showfliers=True,
    )
