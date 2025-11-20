import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
from datasets import (
    Dataset
)
from loguru import logger
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from statsmodels.stats.multitest import multipletests

FORMAT_DICT = {
    "WER": "WER",
    "CER": "CER",
    "mean_pitch_hz": "Mean Pitch (Hz)",
    "median_pitch_hz": "Median Pitch (Hz)",
    "voiced_ratio": "Voiced Ratio",
    "word_rate": "Word Rate",
    "word_count": "Word Count",
    "semantic_distance": "SemDist",
    "loudness": "Loudness (dB)",
    "clip_length": "Clip Length (s)",
    "dataset_name": "Dataset",
    "model": "Model",
    "coral-v2": "CoRal v2",
    "fleurs": "Fleurs",
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
    "dialect_group": "Dialect Group",
    "age_group": "Age Group",
}

# --- Dialect-to-group mapping ---
SUB_DIALECT_TO_DIALECT = {
    "midtøstjysk": "Østjysk",
    "østjysk": "Østjysk",
    "amagermål": "Københavnsk",
    "nørrejysk": "Nordjysk",
    "vestjysk": "Vestjysk",
    "nordsjællandsk": "Sjællandsk",
    "sjællandsk": "Sjællandsk",
    "fynsk": "Fynsk",
    "bornholmsk": "Bornholmsk",
    "sønderjysk": "Sønderjysk",
    "vendsysselsk (m. hanherred og læsø)": "Nordjysk",
    "østligt sønderjysk (m. als)": "Sønderjysk",
    "nordvestsjællandsk": "Sjællandsk",
    "thybomål": "Vestjysk",
    "himmerlandsk": "Nordjysk",
    "djurslandsk (nord-, syddjurs m. nord- og sydsamsø, anholt)": "Østjysk",
    "sydsjællandsk (sydligt sydsjællandsk)": "Sjællandsk",
    "sydfynsk": "Fynsk",
    "morsingmål": "Vestjysk",
    "sydøstjysk": "Østjysk",
    "østsjællandsk": "Sjællandsk",
    "syd for rigsgrænsen: mellemslesvisk, angelmål, fjoldemål": "Sønderjysk",
    "vestfynsk (nordvest-, sydvestfynsk)": "Fynsk",
    "vestlig sønderjysk (m. mandø og rømø)": "Sønderjysk",
    "sydvestjysk (m. fanø)": "Vestjysk",
    "sallingmål": "Vestjysk",
    "nordfalstersk": "Sydømål",
    "langelandsk": "Fynsk",
    "sydvestsjællandsk": "Sjællandsk",
    "lollandsk": "Sydømål",
    "sydømål": "Sydømål",
    "ommersysselsk": "Østjysk",
    "sydfalstersk": "Sydømål",
    "fjandbomål": "Vestjysk",
    "Non-native": "Non-native",
}


def get_top_n_wer_samples(
        df: pd.DataFrame,
        dataset_names: List[str],
        models: List[str],
        top_n: int = 10
    ) -> Dict[str, Dict[str, float]]:
    """
    Get the top N samples with the highest WER for each model and dataset.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation results with 'model', 'dataset_name', 'id', and 'WER' columns.
        dataset_names (List[str]): List of dataset names to consider.
        models (List[str]): List of model names to consider.
        top_n (int): Number of top samples to retrieve per model and dataset.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with model names as keys and dictionaries of sample IDs to WER as values.
    """
    sample_ids_coral = {}
    sample_ids_fleurs = {}
    for dataset_name in dataset_names:
        for model in models:
            model_samples = df[(df["model"] == model) & (df["dataset_name"] == dataset_name)]
            top_samples = model_samples.nlargest(top_n, "WER")
            sample_id_to_wer = {row["id"]: row["WER"] for _, row in top_samples.iterrows()}
            if dataset_name == "coral-v2":
                sample_ids_coral[model] = sample_id_to_wer
            elif dataset_name == "fleurs":
                sample_ids_fleurs[model] = sample_id_to_wer
    return {"coral-v2": sample_ids_coral, "fleurs": sample_ids_fleurs}


def get_samples(dataset: Dataset, dataframe: pd.DataFrame, model: str, ids: Dict[str, Dict[str, float]]) -> None:
    """Get a sample from the dataset by its ID.

    Args:
        dataset:
            The dataset to search in.
        sample_id:
            The ID of the sample to retrieve.
        dataframe:
            The DataFrame containing the samples.
    Returns:
        None
    """
    
    logger.info(f"Getting samples for model: {model}")
    dataset_entries = []
    entry_dict = {}
    for sample_id in ids[model]:
        for entry in dataset:
            if entry["id_recording"] == sample_id:
                dataset_entries.append(entry)
                break
    for i, sample_id in enumerate(ids[model]):
        df_entry = dataframe[(dataframe["id"] == sample_id) & (dataframe["model"] == model)]
        wer = df_entry["WER"].values[0]
        sem_dist = df_entry["semantic_distance"].values[0]
        prediction = df_entry["prediction"].values[0]
        label = df_entry["label"].values[0]
        # add to dict
        entry_dict[i] = {
            "id_recording": sample_id,
            "WER": wer,
            "Semantic Distance": sem_dist,
            "Prediction": prediction,
            "Label": label
        }
    
    df_samples = pd.DataFrame.from_dict(entry_dict, orient="index")
    return df_samples


def star_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'
    

def _fmt(format_dict: Dict[str, str], label: str) -> str:
    return format_dict.get(label, label)
    

def spearman_correlation_plot(
    df_filtered: pd.DataFrame,
    model: str,
    dataset: str,
    format_dict: Dict[str, str],
    target_metrics: List[str],
    feature_metrics: List[str],
    alpha: float = 0.05,
) -> None:
    """
    Generate and save Spearman correlation heatmaps between target metrics and feature metrics
    for each model and dataset combination.
    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame containing evaluation results.
        model_names (Dict[str, str]): Mapping of model identifiers to display names.
    """
    df_md = df_filtered[(df_filtered["model"] == model) & (df_filtered["dataset_name"] == dataset)]
    if len(df_md) < 2:
        return  # not enough data for correlation

    valid_targets = [m for m in target_metrics if m in df_md.columns and df_md[m].notna().sum() > 1]
    valid_features = [m for m in feature_metrics if m in df_md.columns and df_md[m].notna().sum() > 1]
    if not valid_targets or not valid_features:
        return  # not enough valid metrics for correlation

    subset = valid_targets + valid_features
    corr = df_md[subset].corr(method="spearman")
    corr_rect = corr.loc[valid_targets, valid_features]

    plt.figure(figsize=(8, 4 + 0.4 * len(valid_targets)))
    ax = sns.heatmap(
        corr_rect,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        xticklabels=[_fmt(format_dict, m) for m in valid_features],
        yticklabels=[_fmt(format_dict, m) for m in valid_targets],
        cbar_kws={"label": "Spearman $\\rho$"},
    )
    plt.title(f"Spearman Correlation: Sentence Measures vs Acoustic Features\nModel: {_fmt(format_dict, model)}, Dataset: {_fmt(format_dict, dataset)}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # gather all p values for this matrix
    pairs = []
    pvals = []
    ns_for_pair = {}  # optional: sample size used in each test

    for target in valid_targets:
        for feature in valid_features:
            x = df_md[target]
            y = df_md[feature]
            # pairwise drop NaNs to match Spearman behavior
            xy = pd.concat([x, y], axis=1).dropna()
            ns_for_pair[(target, feature)] = len(xy)
            if len(xy) < 3:
                p = np.nan
            else:
                rho, p = stats.spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
            pairs.append((target, feature))
            pvals.append(p)

    # FDR correction on non NaN p values
    pvals = np.array(pvals, dtype=float)
    mask_valid = ~np.isnan(pvals)
    pvals_adj = np.full_like(pvals, np.nan, dtype=float)

    if mask_valid.sum() > 0:
        _, p_corrected, _, _ = multipletests(pvals[mask_valid], alpha=alpha, method='fdr_bh')
        pvals_adj[mask_valid] = p_corrected

    p_adj_dict = {pairs[i]: pvals_adj[i] for i in range(len(pairs))}

    # annotate stars from adjusted p values
    for i, target in enumerate(valid_targets):
        for j, feature in enumerate(valid_features):
            p_adj = p_adj_dict[(target, feature)]
            tag = 'NA' if np.isnan(p_adj) else star_from_p(p_adj)
            # put stars in top right corner of each cell
            ax.text(j + 0.85, i + 0.25, tag, color='black', ha='right', va='center', fontsize=11)

    # optional: add a caption about FDR
    ax.figure.text(0.5, -0.02,
                "Stars reflect Benjamini-Hochberg FDR-adjusted two-sided p-values per model dataset matrix.",
                ha='center', va='top', fontsize=9)
    save_path = Path("reports/plots/deep_analysis/")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"spearman_correlation_{model}_{dataset}.png", bbox_inches='tight')
    plt.close()


def epsilon_squared(H, n, k):
    return (H - k + 1) / (n - k)

def kruskal_wallis(
    df: pd.DataFrame,
    model_name: str,
    format_dict: Dict[str, str],
    group_col: str,
) -> None:
    """Perform Kruskal-Wallis test and Dunn post-hoc analysis on WER across dialect groups for CoRal-v2 dataset."""
    print("\n" + "="*80)
    print(f"Model: {model_name}")
    print("="*80)
    
    # Skip if not enough data
    if df[group_col].nunique() < 3:
        print(f"Not enough {_fmt(format_dict, group_col)} for statistical test.")
        return
    
    # Run Kruskal–Wallis
    groups = [g["WER"].to_numpy() for _, g in df.groupby(group_col)]
    H, p = kruskal(*groups)
    eps2 = epsilon_squared(H, len(df), df[group_col].nunique())
    print(f"Kruskal-Wallis: H = {H:.3f}, p = {p:.3e}, k = {len(groups)}")
    print(f"Epsilon squared ≈ {eps2:.3f}")
    
    # Dunn post-hoc
    posthoc_grouped = sp.posthoc_dunn(
        df, val_col="WER", group_col=group_col, p_adjust="holm"
    )
    print("\nPairwise adjusted p-values:")
    print(posthoc_grouped)
    
    # Heatmap of p-values
    ph = posthoc_grouped.clip(lower=0.0001)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        ph,
        cmap="viridis_r",
        norm=mcolors.LogNorm(vmin=0.0001, vmax=1),
        cbar_kws={"label": "Adjusted p-value (Holm)"},
        square=True,
        annot=True,
        fmt=".4f",
    )
    plt.title(f"Pairwise Dunn Test: WER Differences by CoRal-v2 {_fmt(format_dict, group_col)} for {_fmt(format_dict, model_name)}", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"reports/plots/deep_analysis/dunn_posthoc_wer_{group_col}_{model_name}_coral_v2.png", bbox_inches='tight')
    plt.close()


def mean_wer_by_group(
    df: pd.DataFrame,
    format_dict: Dict[str, str],
    group_col: str,
) -> None:
    """Plot mean WER by group for CoRal-v2 dataset."""
    # Order dialects by overall mean (across models)
    # only sort for dialect_group
    if group_col == "dialect_group":
        group_order = (
            df.groupby(group_col)["WER"]
            .mean()
            .sort_values()
            .index
        )
    else:
        group_order = (
            df.groupby(group_col)["WER"]
            .mean()
            .index
        )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        y=group_col,
        x="WER",
        hue="model",
        order=group_order,
        orient="h",
        estimator=np.mean,        # mean bars
        errorbar=("se"),       # ± standard error
        capsize=0.25,              # small caps on error bars
        err_kws={"linewidth": 2}, # style of error bars
    )

    plt.title(f"Mean WER by {_fmt(format_dict, group_col)} and Model (CoRal-v2)")
    plt.xlabel("Mean WER")
    plt.ylabel(_fmt(format_dict, group_col))
    plt.legend(title="Model")
    # set the model names in the legend to be full names
    handles, labels = plt.gca().get_legend_handles_labels()
    full_labels = [_fmt(format_dict, label) for label in labels]

    # for each bar, add a marker that has the median WER for that age group and model
    median_df_age = (
        df
        .groupby([group_col, "model"])["WER"]
        .median()
        .reset_index()
    )

    for i, row in median_df_age.iterrows():
        group_value = row[group_col]
        model = row["model"]
        median_wer = row["WER"]
        # find the position of the bar
        group_index = list(group_order).index(group_value)
        model_index = list(df["model"].unique()).index(model)
        # calculate the x position of the bar
        num_models = len(df["model"].unique())
        total_bar_width = 0.8  # default total width for seaborn barplot
        bar_width = total_bar_width / num_models
        x_pos = median_wer
        y_pos = group_index - total_bar_width / 2 + bar_width / 2 + model_index * bar_width
        plt.plot(x_pos, y_pos, marker="D", color="black", markersize=6, label="Median WER" if i == 0 else "")

    plt.legend(handles, full_labels, title="Model")
    plt.tight_layout()
    plt.savefig(f"reports/plots/deep_analysis/mean_wer_by_{group_col}_and_model_coral_v2.png", bbox_inches='tight')
    plt.close()
