from pathlib import Path

from datasets import Dataset
from loguru import logger
import pandas as pd

from utils.deep_evaluation_analysis_utils_csr import (
    FORMAT_DICT,
    SUB_DIALECT_TO_DIALECT,
    get_samples,
    get_top_n_wer_samples,
    kruskal_wallis,
    mean_wer_by_group,
    mean_wer_by_group_bootstrapped,
    mean_semdist_by_group_bootstrapped,
    spearman_correlation_plot,
    spearman_correlation_plot_pair
)

from utils.evaluation_utils import load_from_parquet

from utils.manifest_to_hf import manifest_to_hf_dataset

# ==============================
# Configuration
# ==============================

TARGET_METRICS = [
    "WER",
#    "CER",
    "semantic_distance"
]

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
    "canary-finetune_spec-aug_speed-perturbations",
    "parakeet-finetune_spec-aug",
    # "canary-finetune_SA_SP_ll",
    "canary-finetune_SA_SP_ll_SA",
    # "canary-finetune_SA_SP_ll_PS",
    # "canary-finetune_SA_SP_ll_SP",
    # "canary-finetune_SA_SP_ll_SA_PS",
    # "canary-finetune_SA_SP_ll_SA_SP",
    # "canary-finetune_SA_SP_ll_PS_SP",
    # "canary-finetune_SA_SP_ll_SA_PS_SP",
    # "parakeet-finetune_SA_ll",
    # "parakeet-finetune_SA_ll_SA",
    # "parakeet-finetune_SA_ll_PS",
    # "parakeet-finetune_SA_ll_SP",
    #"parakeet-finetune_SA_ll_SA_PS",
     "parakeet-finetune_SA_ll_SA_SP",
    # "parakeet-finetune_SA_ll_PS_SP",
    # "parakeet-finetune_SA_ll_SA_PS_SP",
]

MODEL_FAMILIES = {
    "parakeet": [model for model in FINETUNED_MODELS if "parakeet" in model],
    "canary": [model for model in FINETUNED_MODELS if "canary" in model],
}

ALL_MODELS = BASELINE_MODEL + FINETUNED_MODELS

DATASETS = ["fleurs", "coral-v2", "lillelyd"]

CORAL_GROUPS = [
    "dialect_group",
    "age_group"
]

LILLELYD_GROUPS = [
    "emotion",
    "sentence_id",
]

ALPHA = 0.05


def _get_models(finetuning: bool, all_models: bool = False) -> list[str]:
    if all_models:
        return ALL_MODELS
    if finetuning:
        return FINETUNED_MODELS
    else:
        return BASE_MODELS


def deep_evaluation_analysis_csr(skip_samples: bool, finetuning: bool = False, all_models: bool = False) -> None:
    """Perform deep evaluation analysis by loading evaluation data, getting top WER samples, and generating correlation plots."""

    logger.info("Loading evaluation data...")
    df = load_from_parquet(
        Path("reports/metrics/lillelyd_finetune_stitched_detailed_results.parquet")
    )

    MODELS = _get_models(finetuning=finetuning, all_models=all_models)

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
            elif dataset_name == "lillelyd":
                dataset = manifest_to_hf_dataset(
                    data_dir="data/processed/LilleLyd", 
                    manifest_path=Path("data/processed/LilleLyd/manifest.jsonl")
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
                # target_metrics=TARGET_METRICS,
                target_metrics=FEATURE_METRICS,
                # feature_metrics=FEATURE_METRICS,
                feature_metrics=TARGET_METRICS,
                model=model,
                dataset=dataset,
                format_dict=FORMAT_DICT,
                save_path=Path("reports/csr_finetuning_deep_analysis/")
            )

    for dataset in DATASETS:
        for family, family_models in MODEL_FAMILIES.items():
            logger.info(
                f"Generating Spearman correlation plot for family {family} on dataset {dataset}..."
            )
            spearman_correlation_plot_pair(
                df_filtered=df_filtered,
                target_metrics=FEATURE_METRICS,
                feature_metrics=TARGET_METRICS,
                model_left=family_models[0],
                model_right=family_models[-1],
                dataset=dataset,
                format_dict=FORMAT_DICT,
                save_path=Path("reports/csr_finetuning_deep_analysis/")
            )

    # Focus on CoRal-v2
    logger.info("Focusing analysis on CoRal-v2 dataset...")
    # --- Filter to CoRal ---
    df_filtered_coral = df_filtered[df_filtered["dataset_name"].str.lower() == "coral-v2"].copy()
    # Sanity-check columns
    assert "country_birth" in df_filtered_coral.columns, "country_birth column not found in DataFrame."

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

    # Change age to numeric
    df_filtered_coral["age"] = pd.to_numeric(df_filtered_coral["age"], errors='coerce')

    df_filtered_coral["age_group"] = pd.cut(
        df_filtered_coral["age"], bins=age_bins, labels=age_labels, right=True, include_lowest=True
    )

    for group_col in CORAL_GROUPS:
        logger.info(f"Generating mean WER by {group_col} and model plot for CoRal-v2 dataset...")
        #mean_wer_by_group(df=df_filtered_coral, group_col=group_col, format_dict=FORMAT_DICT)
        mean_wer_by_group_bootstrapped(
            df=df_filtered_coral, 
            dataset_name="coral_v2",
            group_col=group_col, 
            format_dict=FORMAT_DICT,
            save_path=Path("reports/csr_finetuning_deep_analysis/")
        )
        logger.info(f"Generating mean Semantic Distance by {group_col} and model plot for CoRal-v2 dataset...")
        mean_semdist_by_group_bootstrapped(
            df=df_filtered_coral, 
            dataset_name="coral_v2",
            group_col=group_col, 
            format_dict=FORMAT_DICT,
            save_path=Path("reports/csr_finetuning_deep_analysis/")
        )
        for model_name, df_model in df_filtered_coral.groupby("model"):
            logger.info(f"Kruskal-Wallis test for model {model_name} on CoRal-v2 dataset...")
            kruskal_wallis(
                df=df_model, 
                dataset_name="coral_v2",
                model_name=model_name, 
                format_dict=FORMAT_DICT, 
                group_col=group_col,
                save_path=Path("reports/csr_finetuning_deep_analysis/")
            )

    # Focus on LilleLyd
    logger.info("Focusing analysis on LilleLyd dataset...")
    # --- Filter to LilleLyd ---
    df_filtered_lillelyd = df_filtered[df_filtered["dataset_name"].str.lower() == "lillelyd"].copy()

    # Rename emotion labels if needed
    emotion_mapping = {
        "anger": "Anger",
        "neutral": "Neutral",
        "boredom": "Boredom",
        "happiness": "Happiness",
        "sadness": "Sadness",
    }
    df_filtered_lillelyd["emotion"] = df_filtered_lillelyd["emotion"].map(emotion_mapping)

    # Map text to simpler labels
    text_mapping = {
        "Dugen ligger på køleskabet.": "1",
        "Det sorte ark papir er placeret deroppe ved siden af tømmerstykket.": "2",
        "De bar det ovenpå og nu skal de ned igen.": "3",
        "det vil være på det sted, hvor vi altid opbevarer det.": "4",
        "Om syv timer er det morgen.": "5",
    }
    df_filtered_lillelyd["sentence_id"] = df_filtered_lillelyd["text"].map(text_mapping)
    
    for group_col in LILLELYD_GROUPS:
        logger.info(f"Generating mean WER by {group_col} and model plot for LilleLyd dataset...")
        #mean_wer_by_group(df=df_filtered_lillelyd, group_col=group_col, format_dict=FORMAT_DICT)
        mean_wer_by_group_bootstrapped(
            df=df_filtered_lillelyd, 
            dataset_name="lillelyd",
            group_col=group_col, 
            format_dict=FORMAT_DICT,
            save_path=Path("reports/csr_finetuning_deep_analysis/")
        )
        logger.info(f"Generating mean Semantic Distance by {group_col} and model plot for LilleLyd dataset...")
        mean_semdist_by_group_bootstrapped(
            df=df_filtered_lillelyd, 
            dataset_name="lillelyd",
            group_col=group_col, 
            format_dict=FORMAT_DICT,
            save_path=Path("reports/csr_finetuning_deep_analysis/")
        )
        for model_name, df_model in df_filtered_lillelyd.groupby("model"):
            logger.info(f"Kruskal-Wallis test for model {model_name} on LilleLyd dataset on {group_col}...")
            kruskal_wallis(
                df=df_model, 
                dataset_name="lillelyd",
                model_name=model_name, 
                format_dict=FORMAT_DICT, 
                group_col=group_col,
                save_path=Path("reports/csr_finetuning_deep_analysis/")
            )
