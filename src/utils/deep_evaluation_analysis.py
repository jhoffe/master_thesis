from pathlib import Path

from datasets import Dataset
from loguru import logger
import pandas as pd

from utils.deep_evaluation_analysis_utils import (
    FORMAT_DICT,
    SUB_DIALECT_TO_DIALECT,
    get_samples,
    get_top_n_wer_samples,
    kruskal_wallis,
    mean_wer_by_group,
    mean_wer_by_group_bootstrapped,
    spearman_correlation_plot,
)
from utils.evaluation_utils import load_from_parquet

# ==============================
# Configuration
# ==============================

TARGET_METRICS = ["WER", "CER", "semantic_distance"]
FEATURE_METRICS = [
    "mean_pitch_hz",
    "median_pitch_hz",
    "voiced_ratio",
    "word_rate",
    "word_count",
    "loudness",
    "clip_length",
]

BASELINE_MODEL = ["roest-whisper-large-v1"]

BASE_MODELS = ["roest-whisper-large-v1", "parakeet-tdt-0.6b-v3", "canary-1b-v2"]

FINETUNED_MODELS = [
    #"parakeet_finetune",
    #"parakeet_finetune_pitch-shift",
    "parakeet_finetune_spec-aug",
    #"parakeet_finetune_speed-perturbations",
    #"parakeet_finetune_spec-aug_pitch-shift",
    "parakeet_finetune_spec-aug_speed-perturbations",
    #"parakeet_finetune_speed-perturbations_pitch-shift",
    #"parakeet_finetune_spec-aug_speed-perturbations_pitch-shift",
    #"canary_finetune",
    #"canary_finetune_pitch-shift",
    #"canary_finetune_spec-aug",
    #"canary_finetune_speed-perturbations",
    #"canary_finetune_spec-aug_pitch-shift",
    "canary_finetune_spec-aug_speed-perturbations",
    #"canary_finetune_speed-perturbations_pitch-shift",
    #"canary_finetune_spec-aug_speed-perturbations_pitch-shift",
]

ALL_MODELS = BASELINE_MODEL + FINETUNED_MODELS

DATASETS = ["fleurs", "coral-v2"]

GROUPS = [
    "dialect_group",
    "age_group",
]

ALPHA = 0.05


def _get_models(finetuning: bool, all_models: bool = False) -> list[str]:
    if all_models:
        return ALL_MODELS
    if finetuning:
        return FINETUNED_MODELS
    else:
        return BASE_MODELS


def deep_evaluation_analysis(skip_samples: bool, finetuning: bool = False, all_models: bool = False) -> None:
    """Perform deep evaluation analysis by loading evaluation data, getting top WER samples, and generating correlation plots."""

    logger.info("Loading evaluation data...")
    df = load_from_parquet(
        Path("reports/metrics/combined_detailed_results_with_embeddings.parquet")
    )

    MODELS = _get_models(finetuning, all_models=all_models)

    logger.info(f"Using models: {MODELS}")

    logger.info("Filtering evaluation data to specified grid...")
    df_filtered = df[df["model"].isin(MODELS) & df["dataset_name"].isin(DATASETS)]

    if not skip_samples:
        logger.info("Getting top 10 WER samples for each model on each dataset...")
        top_samples = get_top_n_wer_samples(
            df=df_filtered,
            dataset_names=DATASETS,
            models=MODELS,
            top_n=10,
        )            

        for dataset_name in DATASETS:
            logger.info(f"Loading dataset: {dataset_name}...")
            if dataset_name == "coral-v2":
                dataset = Dataset.load_from_disk(
                    "data/huggingface/datasets/test-sets/CoRal-project--coral-v2-read_aloud-test-unfiltered"
                )
            elif dataset_name == "fleurs":
                dataset = Dataset.load_from_disk(
                    "data/huggingface/datasets/test-sets/google--fleurs-da_dk-test-unfiltered"
                )
            ids = top_samples[dataset_name]
            for model in MODELS:
                df_samples = get_samples(
                    dataset=dataset,
                    dataframe=df_filtered,
                    model=model,
                    ids=ids,
                )
                # save to parquet
                save_dir = Path(f"reports/samples/{dataset_name}")
                save_dir.mkdir(parents=True, exist_ok=True)
                output_path = Path(f"{save_dir}/{model}_top_wer_samples.parquet")
                df_samples.to_parquet(output_path)

                logger.info(
                    f"Saved top WER samples for model {model} on dataset {dataset_name} to {output_path}"
                )

    for dataset in DATASETS:
        for model in MODELS:
            logger.info(
                f"Generating Spearman correlation plot for model {model} on dataset {dataset}..."
            )
            spearman_correlation_plot(
                df_filtered=df_filtered,
                target_metrics=TARGET_METRICS,
                feature_metrics=FEATURE_METRICS,
                model=model,
                dataset=dataset,
                format_dict=FORMAT_DICT,
                save_path=Path("reports/finetuning_plots/deep_analysis/") if finetuning else Path("reports/plots/deep_analysis/"),
            )

    # Focus on CoRal-v2
    logger.info("Focusing analysis on CoRal-v2 dataset...")
    # --- Filter to CoRal ---
    df_filtered_coral = df_filtered[df_filtered["dataset_name"].str.lower() == "coral-v2"].copy()
    # Add the "Non-native" dialect label
    df_filtered_coral.loc[df_filtered_coral["country_birth"] != "DK", "dialect"] = "Non-native"

    # Apply mapping
    df_filtered_coral["dialect_group"] = df_filtered_coral["dialect"].map(SUB_DIALECT_TO_DIALECT)
    df_filtered_coral = df_filtered_coral.dropna(subset=["dialect_group"])

    # Add age groups
    # --- Age binning setup ---
    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    age_labels = [
        "<20",
        "21 to 30",
        "31 to 40",
        "41 to 50",
        "51 to 60",
        "61 to 70",
        "71 to 80",
        "81<",
    ]

    df_filtered_coral["age_group"] = pd.cut(
        df_filtered_coral["age"], bins=age_bins, labels=age_labels, right=True, include_lowest=True
    )

    for group_col in GROUPS:
        logger.info(f"Generating mean WER by {group_col} and model plot for CoRal-v2 dataset...")
        #mean_wer_by_group(df=df_filtered_coral, group_col=group_col, format_dict=FORMAT_DICT)
        mean_wer_by_group_bootstrapped(
            df=df_filtered_coral, 
            group_col=group_col, 
            format_dict=FORMAT_DICT,
            save_path=Path("reports/finetuning_plots/deep_analysis/") if finetuning else Path("reports/plots/deep_analysis/"),
        )
        for model_name, df_model in df_filtered_coral.groupby("model"):
            logger.info(f"Kruskal-Wallis test for model {model_name} on CoRal-v2 dataset...")
            kruskal_wallis(
                df=df_model, 
                model_name=model_name, 
                format_dict=FORMAT_DICT, 
                group_col=group_col,
                save_path=Path("reports/finetuning_plots/deep_analysis/") if finetuning else Path("reports/plots/deep_analysis/"),
            )
