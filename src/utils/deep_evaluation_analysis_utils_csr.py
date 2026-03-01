from pathlib import Path
from typing import Dict, List, Union

from datasets import Dataset
from loguru import logger
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import kruskal
import seaborn as sns
from statsmodels.stats.multitest import multipletests

_FORMAT_DICT: dict[Union[str, tuple[str, ...]], str] = {
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
    ("coral", "coral-v2"): "CoRal-v2",
    "fleurs": "FLEURS",
    "lillelyd": "LilleLyd",
    "roest-whisper-large-v1": "Røst-Whisper",
    ("parakeet", "parakeet-tdt-0.6b-v3"): "Parakeet-TDT",
    ("canary", "canary-1b-v2"): "Canary-1B",
    "dialect_group": "Dialect Group",
    "age_group": "Age Group",
    "emotion": "Emotion",
    "sentence_id": "Sentence No.",
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
    # LONG ABBREVIATIONS
    # # "canary-finetune_spec-aug_speed-perturbations": "Canary-1B-Pre_FT",
    # "canary-finetune_SA_SP_ll": "Canary-1B-FT",
    # # "canary-finetune_SA_SP_ll_SA": "Canary-1B-FT+SA",
    # "canary-finetune_SA_SP_ll_PS": "Canary-1B-FT+PS",
    # "canary-finetune_SA_SP_ll_SP": "Canary-1B-FT+SP",
    # "canary-finetune_SA_SP_ll_SA_PS": "Canary-1B-FT+SA+PS",
    # "canary-finetune_SA_SP_ll_SA_SP": "Canary-1B-FT+SA+SP",
    # "canary-finetune_SA_SP_ll_PS_SP": "Canary-1B-FT+PS+SP",
    # "canary-finetune_SA_SP_ll_SA_PS_SP": "Canary-1B-FT+SA+PS+SP",
    # # "parakeet-finetune_spec-aug": "Parakeet-TDT-Pre_FT",
    # "parakeet-finetune_SA_ll": "Parakeet-TDT-FT",
    # "parakeet-finetune_SA_ll_SA": "Parakeet-TDT-FT+SA",
    # "parakeet-finetune_SA_ll_PS": "Parakeet-TDT-FT+PS",
    # "parakeet-finetune_SA_ll_SP": "Parakeet-TDT-FT+SP",
    # "parakeet-finetune_SA_ll_SA_PS": "Parakeet-TDT-FT+SA+PS",
    # "parakeet-finetune_SA_ll_SA_SP": "FT (SA+SP)",
    # "parakeet-finetune_SA_ll_PS_SP": "Parakeet-TDT-FT+PS+SP",
    # "parakeet-finetune_SA_ll_SA_PS_SP": "Parakeet-TDT-FT+SA+PS+SP",
    # SHORT ABBREVIATIONS
    # "canary-finetune_spec-aug_speed-perturbations": "Canary SA+SP",
    "canary-finetune_spec-aug_speed-perturbations": "Baseline", # For Correlation Analysis
    "canary-finetune_SA_SP_ll": "Canary SA+SP / -",
    # "canary-finetune_SA_SP_ll_SA": "Canary SA+SP / SA",
    "canary-finetune_SA_SP_ll_SA": "FT (SA)", # For Correlation Analysis
    "canary-finetune_SA_SP_ll_PS": "Canary SA+SP / PS",
    "canary-finetune_SA_SP_ll_SP": "Canary SA+SP / SP",
    "canary-finetune_SA_SP_ll_SA_PS": "Canary SA+SP / SA+PS",
    "canary-finetune_SA_SP_ll_SA_SP": "Canary SA+SP / SA+SP",
    "canary-finetune_SA_SP_ll_PS_SP": "Canary SA+SP / PS+SP",
    "canary-finetune_SA_SP_ll_SA_PS_SP": "Canary SA+SP / SA+PS+SP",
    # "parakeet-finetune_spec-aug": "Parakeet SA",
    "parakeet-finetune_spec-aug": "Baseline", # For Correlation Analysis
    "parakeet-finetune_SA_ll": "Parakeet SA / -",
    "parakeet-finetune_SA_ll_SA": "Parakeet SA / SA",
    "parakeet-finetune_SA_ll_PS": "Parakeet SA / PS",
    "parakeet-finetune_SA_ll_SP": "Parakeet SA / SP",
    "parakeet-finetune_SA_ll_SA_PS": "Parakeet SA / SA+PS",
    # "parakeet-finetune_SA_ll_SA_SP": "Parakeet SA / SA+SP",
    "parakeet-finetune_SA_ll_SA_SP": "FT (SA+SP)", # For Correlation Analysis
    "parakeet-finetune_SA_ll_PS_SP": "Parakeet SA / PS+SP",
    "parakeet-finetune_SA_ll_SA_PS_SP": "Parakeet SA / SA+PS+SP",
    "lillelyd cv-1": "LilleLyd CV-1",
    "lillelyd cv-2": "LilleLyd CV-2",
    "lillelyd cv-3": "LilleLyd CV-3",
    "lillelyd cv-4": "LilleLyd CV-4",
    "coral-v2 averaged": "CoRal-v2 Averaged",
    "fleurs averaged": "FLEURS Averaged",
    "anger": "Anger",
    "neutral": "Neutral",
    "boredom": "Boredom",
    "happiness": "Happiness",
    "sadness": "Sadness",
    "Dugen ligger på køleskabet.": "1",
    "Det sorte ark papir er placeret deroppe ved siden af tømmerstykket.": "2",
    "De bar det ovenpå og nu skal de ned igen.": "3",
    "det vil være på det sted, hvor vi altid opbevarer det.": "4",
    "Om syv timer er det morgen.": "5",
}


def make_format_dict(format_dict: dict[Union[str, tuple[str, ...]], str]) -> dict[str, str]:
    """Expand FORMAT_DICT to map each individual string in tuple keys to the same value.

    Args:
        format_dict (dict[Union[str, Sequence[str]], str]): Original format dictionary with string or tuple keys.

    Returns:
        dict[str, str]: Expanded format dictionary with only string keys.
    """
    expanded_dict = {}
    for key, value in format_dict.items():
        if isinstance(key, tuple):
            for sub_key in key:
                expanded_dict[sub_key] = value
        else:
            expanded_dict[key] = value
    return expanded_dict


def format(label: str) -> str:
    return FORMAT_DICT.get(label, label)


FORMAT_DICT = make_format_dict(_FORMAT_DICT)

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
    df: pd.DataFrame, dataset_names: List[str], models: List[str], top_n: int = 10
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
    sample_ids_lillelyd = {}
    for dataset_name in dataset_names:
        for model in models:
            model_samples = df[(df["model"] == model) & (df["dataset_name"] == dataset_name)]
            if model_samples.empty:
                continue
            top_samples = model_samples.nlargest(top_n, "WER")
            sample_id_to_wer = {row["id"]: row["WER"] for _, row in top_samples.iterrows()}
            if dataset_name == "coral-v2":
                sample_ids_coral[model] = sample_id_to_wer
            elif dataset_name == "fleurs":
                sample_ids_fleurs[model] = sample_id_to_wer
            elif dataset_name == "lillelyd":
                    sample_ids_lillelyd[model] = sample_id_to_wer
    return {"coral-v2": sample_ids_coral, "fleurs": sample_ids_fleurs, "lillelyd": sample_ids_lillelyd}


def get_samples(
    dataset: Dataset, dataframe: pd.DataFrame, model: str, ids: Dict[str, Dict[str, float]]
) -> None:
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
            "Label": label,
        }

    df_samples = pd.DataFrame.from_dict(entry_dict, orient="index")
    return df_samples


def star_from_p(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


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
    save_path: Path = Path("reports/csr_finetuning_plots/deep_analysis/")
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

    valid_targets = [
        m for m in target_metrics if m in df_md.columns and df_md[m].notna().sum() > 1
    ]
    valid_features = [
        m for m in feature_metrics if m in df_md.columns and df_md[m].notna().sum() > 1
    ]
    if not valid_targets or not valid_features:
        return  # not enough valid metrics for correlation

    subset = valid_targets + valid_features
    corr = df_md[subset].corr(method="spearman")
    corr_rect = corr.loc[valid_targets, valid_features]

    plt.figure(figsize=(4, 6))
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
    # plt.title(
    #     f"Spearman Correlation: Sentence Measures vs Acoustic Features\nModel: {_fmt(format_dict, model)}, Dataset: {_fmt(format_dict, dataset)}"
    # )
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
        _, p_corrected, _, _ = multipletests(pvals[mask_valid], alpha=alpha, method="fdr_bh")
        pvals_adj[mask_valid] = p_corrected

    p_adj_dict = {pairs[i]: pvals_adj[i] for i in range(len(pairs))}

    # annotate stars from adjusted p values
    for i, target in enumerate(valid_targets):
        for j, feature in enumerate(valid_features):
            p_adj = p_adj_dict[(target, feature)]
            tag = "NA" if np.isnan(p_adj) else star_from_p(p_adj)
            # put stars in top right corner of each cell
            ax.text(j + 0.85, i + 0.25, tag, color="black", ha="right", va="center", fontsize=11)

    # optional: add a caption about FDR
    # ax.figure.text(
    #     0.5,
    #     -0.02,
    #     "Stars reflect Benjamini-Hochberg FDR-adjusted two-sided p-values per model dataset matrix.",
    #     ha="center",
    #     va="top",
    #     fontsize=9,
    # )
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"spearman_correlation_{model}_{dataset}.png", bbox_inches="tight")
    plt.close()


def _spearman_corr_and_pvals(
    df_md: pd.DataFrame,
    valid_targets: List[str],
    valid_features: List[str],
    alpha: float,
):
    subset = valid_targets + valid_features
    corr = df_md[subset].corr(method="spearman")
    corr_rect = corr.loc[valid_targets, valid_features]

    pairs, pvals = [], []
    for target in valid_targets:
        for feature in valid_features:
            xy = df_md[[target, feature]].dropna()
            if len(xy) < 3:
                p = np.nan
            else:
                _, p = stats.spearmanr(xy[target], xy[feature])
            pairs.append((target, feature))
            pvals.append(p)

    pvals = np.array(pvals, dtype=float)
    mask_valid = ~np.isnan(pvals)
    pvals_adj = np.full_like(pvals, np.nan, dtype=float)

    if mask_valid.sum() > 0:
        _, p_corrected, _, _ = multipletests(pvals[mask_valid], alpha=alpha, method="fdr_bh")
        pvals_adj[mask_valid] = p_corrected

    p_adj_dict = {pairs[i]: pvals_adj[i] for i in range(len(pairs))}
    return corr_rect, p_adj_dict


def spearman_correlation_plot_pair(
    df_filtered: pd.DataFrame,
    model_left: str,
    model_right: str,
    dataset: str,
    format_dict: Dict[str, str],
    target_metrics: List[str],
    feature_metrics: List[str],
    alpha: float = 0.05,
    save_path: Path = Path("reports/plots/deep_analysis/"),
) -> None:

    def _prep(model: str):
        df_md = df_filtered[(df_filtered["model"] == model) & (df_filtered["dataset_name"] == dataset)]
        if len(df_md) < 2:
            return None, None, None, None

        valid_targets = [m for m in target_metrics if m in df_md.columns and df_md[m].notna().sum() > 1]
        valid_features = [m for m in feature_metrics if m in df_md.columns and df_md[m].notna().sum() > 1]
        if not valid_targets or not valid_features:
            return None, None, None, None

        corr_rect, p_adj_dict = _spearman_corr_and_pvals(
            df_md=df_md,
            valid_targets=valid_targets,
            valid_features=valid_features,
            alpha=alpha,
        )
        return corr_rect, p_adj_dict, valid_targets, valid_features

    left = _prep(model_left)
    right = _prep(model_right)
    if left[0] is None or right[0] is None:
        return

    corr_left, p_left, vt_left, vf_left = left
    corr_right, p_right, vt_right, vf_right = right

    # Enforce identical row/col order. If they differ, intersect and align.
    vt = [m for m in vt_left if m in vt_right]
    vf = [m for m in vf_left if m in vf_right]
    if len(vt) == 0 or len(vf) == 0:
        return

    corr_left = corr_left.loc[vt, vf]
    corr_right = corr_right.loc[vt, vf]

    # Build figure with fixed layout: left heatmap, right heatmap, one shared colorbar
    fig = plt.figure(figsize=(4, 6), dpi=150)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.1], wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # Common tick labels
    xt = [_fmt(format_dict, m) for m in vf]
    yt = [_fmt(format_dict, m) for m in vt]

    sns.heatmap(
        corr_left,
        ax=ax1,
        cbar=True,
        cbar_ax=cax,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        xticklabels=xt,
        yticklabels=yt,
        cbar_kws={"label": "Spearman $\\rho$"},
    )

    sns.heatmap(
        corr_right,
        ax=ax2,
        cbar=False,  # only one shared colorbar
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        xticklabels=xt,
        yticklabels=yt,
    )

    # Titles per panel (optional but usually useful)
    ax1.set_title(_fmt(format_dict, model_left))
    ax2.set_title(_fmt(format_dict, model_right))

    for ax in (ax1, ax2):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Hide y labels on the right to save space (keeps same size)
    ax2.tick_params(left=False, labelleft=False)

    # Star annotations (top right of each cell)
    for i, target in enumerate(vt):
        for j, feature in enumerate(vf):
            tag1 = star_from_p(p_left.get((target, feature), np.nan))
            tag2 = star_from_p(p_right.get((target, feature), np.nan))
            ax1.text(j + 0.85, i + 0.25, tag1, color="black", ha="right", va="center")
            ax2.text(j + 0.85, i + 0.25, tag2, color="black", ha="right", va="center")

    #fig.suptitle(f"Spearman correlations on {_fmt(format_dict, dataset)}", y=0.98)
    #fig.subplots_adjust(left=0.22, right=0.94, top=0.90, bottom=0.14)

    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"spearman_pair_{model_left}_vs_{model_right}_{dataset}.png", bbox_inches="tight")
    plt.close(fig)


def epsilon_squared(H, n, k):
    return (H - k + 1) / (n - k)


def kruskal_wallis(
    df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    format_dict: Dict[str, str],
    group_col: str,
    save_path: Path = Path("reports/csr_finetuning_plots/deep_analysis/"),
) -> None:
    """Perform Kruskal-Wallis test and Dunn post-hoc analysis on WER across dialect groups for dataset."""
    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print("=" * 80)

    # Skip if not enough data
    if df[group_col].nunique() < 3:
        print(f"Not enough {_fmt(format_dict, group_col)} for statistical test.")
        return

    # Run Kruskal–Wallis
    groups = [g["WER"].to_numpy() for _, g in df.groupby(group_col, observed=True)]
    H, p = kruskal(*groups)
    eps2 = epsilon_squared(H, len(df), df[group_col].nunique())
    print(f"Kruskal-Wallis: H = {H:.3f}, p = {p:.3e}, k = {len(groups)}")
    print(f"Epsilon squared ≈ {eps2:.3f}")
    print(f"Latex:")
    # should be on _fmt{model_name} & $868.01$ & $4.829\mathrm{e}{-181}$ & $0.094$ \\
    print(f"{_fmt(format_dict, model_name)} & ${H:.2f}$ & ${p:.3e}$ & ${eps2:.3f}$ \\\\")

    # Dunn post-hoc
    posthoc_grouped = sp.posthoc_dunn(df, val_col="WER", group_col=group_col, p_adjust="holm")
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
    # plt.title(
    #     f"Pairwise Dunn Test: WER Differences by {_fmt(format_dict, dataset_name)} {_fmt(format_dict, group_col)} for {_fmt(format_dict, model_name)}",
    #     pad=20,
    # )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path / f"dunn_posthoc_wer_{group_col}_{model_name}_{dataset_name}.png",
        bbox_inches="tight",
    )
    plt.close()


def mean_wer_by_group(
    df: pd.DataFrame,
    dataset_name: str,
    format_dict: Dict[str, str],
    group_col: str,
    save_path: Path = Path("reports/csr_finetuning_plots/deep_analysis/"),
) -> None:
    """Plot mean WER by group for dataset."""
    # Order dialects by overall mean (across models)
    # only sort for dialect_group
    if group_col == "dialect_group":
        group_order = df.groupby(group_col, observed=True)["WER"].mean().sort_values().index
    else:
        group_order = df.groupby(group_col, observed=True)["WER"].mean().index

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        y=group_col,
        x="WER",
        hue="model",
        order=group_order,
        orient="h",
        estimator=np.mean,  # mean bars
        errorbar=("se"),  # ± standard error
        capsize=0.25,  # small caps on error bars
        err_kws={"linewidth": 2},  # style of error bars
    )

    # plt.title(f"Mean WER by {_fmt(format_dict, group_col)} and Model for {_fmt(format_dict, dataset_name)}")
    plt.xlabel("Mean WER")
    plt.ylabel(_fmt(format_dict, group_col))
    plt.legend(title="Model")
    # set the model names in the legend to be full names
    handles, labels = plt.gca().get_legend_handles_labels()
    full_labels = [_fmt(format_dict, label) for label in labels]

    # for each bar, add a marker that has the median WER for that age group and model
    median_df_age = df.groupby([group_col, "model"], observed=True)["WER"].median().reset_index()

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
        plt.plot(
            x_pos,
            y_pos,
            marker="D",
            color="black",
            markersize=6,
            label="Median WER" if i == 0 else "",
        )

    plt.legend(handles, full_labels, title="Model")
    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path / f"mean_wer_by_{group_col}_and_model_{dataset_name}.png",
        bbox_inches="tight",
    )
    plt.close()


def _bootstrap_mean_ci(x, level=95, B=5000, rng=None):
    """Percentile bootstrap CI for the mean of 1D array x."""
    x = np.asarray(x)
    n = len(x)
    if rng is None:
        rng = np.random.default_rng()
    boot = np.empty(B, dtype=float)
    for b in range(B):
        sample = rng.choice(x, size=n, replace=True)
        boot[b] = sample.mean()
    alpha = 1 - level / 100
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return x.mean(), low, high


def mean_wer_by_group_bootstrapped(
    df: pd.DataFrame,
    dataset_name: str,
    format_dict: Dict[str, str],
    group_col: str,
    save_path: Path = Path("reports/csr_finetuning_plots/deep_analysis/"),
) -> None:
    """Plot mean WER by group for dataset with bootstrap CIs."""
    rng = np.random.default_rng()

    # Order groups
    if group_col == "dialect_group":
        group_order = df.groupby(group_col, observed=True)["WER"].mean().sort_values().index
    else:
        group_order = df.groupby(group_col, observed=True)["WER"].mean().index

    models = list(df["model"].unique())

    # ---------- bootstrap summary: one row per (group, model) ----------
    summary_rows = []
    for (grp, model), sub in df.groupby([group_col, "model"], observed=True):
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            sub["WER"].to_numpy(), level=95, B=5000, rng=rng
        )
        summary_rows.append(
            {
                group_col: grp,
                "model": model,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary[group_col] = pd.Categorical(summary[group_col], categories=group_order, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], categories=models, ordered=True)
    summary = summary.sort_values([group_col, "model"])

    # Pre-calculate medians
    median_df = df.groupby([group_col, "model"], observed=True)["WER"].median().reset_index()

    # ---------- Dynamic Height Calculation ----------
    # "Wider and shorter" logic
    n_groups = len(group_order)
    n_models = len(models)
    base_bar_height = 0.24 
    calc_height = max(6, n_groups * n_models * base_bar_height + 1.5)

    # Set width to 16
    plt.figure(figsize=(16, calc_height))

    # ---------- barplot of means ----------
    ax = sns.barplot(
        data=summary,
        y=group_col,
        x="mean",
        hue="model",
        order=group_order,
        orient="h",
        errorbar=None,
    )

    # ---------- Combine CIs and Diamonds ----------
    yticks = np.array(ax.get_yticks())
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    median_label_added = False

    for model_index, model in enumerate(models):
        container = ax.containers[model_index]

        for bar in container:
            y_center = bar.get_y() + bar.get_height() / 2
            idx = int(np.argmin(np.abs(yticks - y_center)))
            grp_label = yticklabels[idx]

            # 1. Plot Bootstrap CI
            row = summary[
                (summary[group_col] == grp_label) & 
                (summary["model"] == model)
            ]

            if not row.empty:
                row = row.iloc[0]
                mean = row["mean"]
                ci_low = row["ci_low"]
                ci_high = row["ci_high"]

                ax.errorbar(
                    mean,
                    y_center,
                    xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                    capthick=1.2,
                    linewidth=1.2,
                )

            # 2. Plot Median Diamond
            med_row = median_df[
                (median_df[group_col] == grp_label) & 
                (median_df["model"] == model)
            ]
            
            if not med_row.empty:
                median_val = med_row.iloc[0]["WER"]
                label = ""
                if not median_label_added:
                    label = "Median WER"
                    median_label_added = True

                ax.plot(
                    median_val,
                    y_center,
                    marker="D",
                    color="black",
                    markersize=6,
                    linestyle='None',
                    label=label,
                    zorder=10
                )

    # ---------- STYLING ----------
    # Title included (as present in this function snippet)
    # plt.title(
    #     f"Mean WER by {_fmt(format_dict, group_col)} and Model ({_fmt(format_dict, dataset_name)})", 
    #     fontsize=18
    # )
    plt.xlabel("Mean WER", fontsize=16)
    #plt.ylabel(_fmt(format_dict, group_col), fontsize=16)
    plt.ylabel(_fmt(format_dict, group_col), fontsize=16) if group_col == "sentence_id" else plt.ylabel("")  # blank ylabel for emotion for cleaner look
    
    ax.tick_params(axis='both', which='major', labelsize=16)



    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lbl in labels:
        if lbl == "Median WER":
            new_labels.append(lbl)
        else:
            new_labels.append(_fmt(format_dict, lbl))

    # Conditional Legend Placement
    if group_col == "emotion":
        plt.legend(
            handles=handles,
            labels=new_labels,
            title="Model",               # <--- Added this back
            title_fontsize=16,           # <--- Added this back
            fontsize=16,
            loc="lower center",          # Anchor the bottom of the legend
            bbox_to_anchor=(0.5, 1.02),  # Place it just above the plot
            ncol=len(handles),           # Force single row
            frameon=False                # Optional: looks cleaner without a box on top
        )
    else:
        plt.legend(
            handles, 
            new_labels, 
            title="Model", 
            title_fontsize=16, 
            fontsize=16,
            framealpha=0.95
            # Default placement (usually "best" inside the plot)
        )

    # format x-axis as percentage
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path / f"mean_wer_by_{group_col}_and_model_{dataset_name}.png",
        bbox_inches="tight",
    )
    plt.close()

    
def mean_semdist_by_group_bootstrapped(
    df: pd.DataFrame,
    dataset_name: str,
    format_dict: Dict[str, str],
    group_col: str,
    save_path: Path = Path("reports/csr_finetuning_plots/deep_analysis/"),
) -> None:
    """Plot mean semantic distance by group for dataset with bootstrap CIs."""
    rng = np.random.default_rng()

    # Order groups
    if group_col == "dialect_group":
        group_order = df.groupby(group_col, observed=True)["semantic_distance"].mean().sort_values().index
    else:
        group_order = df.groupby(group_col, observed=True)["semantic_distance"].mean().index

    models = list(df["model"].unique())

    # ---------- bootstrap summary: one row per (group, model) ----------
    summary_rows = []
    for (grp, model), sub in df.groupby([group_col, "model"], observed=True):
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            sub["semantic_distance"].to_numpy(), level=95, B=5000, rng=rng
        )
        summary_rows.append(
            {
                group_col: grp,
                "model": model,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary[group_col] = pd.Categorical(summary[group_col], categories=group_order, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], categories=models, ordered=True)
    summary = summary.sort_values([group_col, "model"])

    # Pre-calculate medians
    median_df = df.groupby([group_col, "model"], observed=True)["semantic_distance"].median().reset_index()

    # ---------- Dynamic Height Calculation ----------
    # Wider (16) and Shorter (0.24 factor) logic
    n_groups = len(group_order)
    n_models = len(models)
    base_bar_height = 0.24 
    calc_height = max(6, n_groups * n_models * base_bar_height + 1.5)

    plt.figure(figsize=(16, calc_height))

    # ---------- barplot of means ----------
    ax = sns.barplot(
        data=summary,
        y=group_col,
        x="mean",
        hue="model",
        order=group_order,
        orient="h",
        errorbar=None,
    )

    # ---------- Combine CIs and Diamonds ----------
    yticks = np.array(ax.get_yticks())
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    median_label_added = False

    for model_index, model in enumerate(models):
        container = ax.containers[model_index]

        for bar in container:
            y_center = bar.get_y() + bar.get_height() / 2
            idx = int(np.argmin(np.abs(yticks - y_center)))
            grp_label = yticklabels[idx]

            # 1. Plot Bootstrap CI
            row = summary[
                (summary[group_col] == grp_label) & 
                (summary["model"] == model)
            ]

            if not row.empty:
                row = row.iloc[0]
                mean = row["mean"]
                ci_low = row["ci_low"]
                ci_high = row["ci_high"]

                ax.errorbar(
                    mean,
                    y_center,
                    xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                    capthick=1.2,
                    linewidth=1.2,
                )

            # 2. Plot Median Diamond
            med_row = median_df[
                (median_df[group_col] == grp_label) & 
                (median_df["model"] == model)
            ]
            
            if not med_row.empty:
                median_val = med_row.iloc[0]["semantic_distance"]
                label = ""
                if not median_label_added:
                    label = "Median SemDist"
                    median_label_added = True

                ax.plot(
                    median_val,
                    y_center,
                    marker="D",
                    color="black",
                    markersize=6,
                    linestyle='None',
                    label=label,
                    zorder=10
                )

    # ---------- STYLING ----------
    # Title with dataset name
    # plt.title(
    #     f"Mean Semantic Distance by {_fmt(format_dict, group_col)} and Model ({_fmt(format_dict, dataset_name)})", 
    #     fontsize=18
    # )
    plt.xlabel("Mean SemDist", fontsize=16)
    plt.ylabel(_fmt(format_dict, group_col), fontsize=16) if group_col == "sentence_id" else plt.ylabel("")  # blank ylabel for emotion for cleaner look

    # Larger Ticks
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Clean Legend Labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lbl in labels:
        if lbl == "Median SemDist":
            new_labels.append(lbl)
        else:
            new_labels.append(_fmt(format_dict, lbl))

    # Conditional Legend Placement
    if group_col == "emotion":
        plt.legend(
            handles=handles,
            labels=new_labels,
            title="Model",               # <--- Added this back
            title_fontsize=16,           # <--- Added this back
            fontsize=16,
            loc="lower center",          # Anchor the bottom of the legend
            bbox_to_anchor=(0.5, 1.02),  # Place it just above the plot
            ncol=len(handles),           # Force single row
            frameon=False                # Optional: looks cleaner without a box on top
        )
    else:
        plt.legend(
            handles, 
            new_labels, 
            title="Model", 
            title_fontsize=16, 
            fontsize=16,
            framealpha=0.95
            # Default placement (usually "best" inside the plot)
        )

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path / f"mean_semantic_distance_by_{group_col}_and_model_{dataset_name}.png",
        bbox_inches="tight",
    )
    plt.close()