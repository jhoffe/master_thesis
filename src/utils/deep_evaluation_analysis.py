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

from IPython.display import Audio, display

from utils.evaluation_utils import (
    load_from_parquet,
)

from utils.deep_evaluation_analysis_utils import (
    get_top_n_wer_samples,
    get_samples,
    spearman_correlation_plot,
    FORMAT_DICT,
    SUB_DIALECT_TO_DIALECT,
)

from statsmodels.stats.multitest import multipletests

# ==============================
# Configuration
# ==============================

TARGET_METRICS = [
    "WER", 
    "CER", "semantic_distance"
    "semantic_distance"
]
FEATURE_METRICS = [
    "mean_pitch_hz", 
    "median_pitch_hz", 
    "voiced_ratio", 
    "word_rate", 
    "word_count", 
    "loudness", 
    "clip_length"
]

MODELS = [
    "roest-whisper-large-v1", 
    "parakeet-tdt-0.6b-v3", 
    "canary-1b-v2"
]

DATASETS = [
    "fleurs",
    "coral-v2"
]

ALPHA = 0.05


def deep_evaluation_analysis() -> None:
    """Perform deep evaluation analysis by loading evaluation data, getting top WER samples, and generating correlation plots."""

    logger.info("Loading evaluation data...")
    df = load_from_parquet(Path("reports/metrics/combined_detailed_results_with_embeddings.parquet"))

    logger.info("Filtering evaluation data to specified grid...")
    df_filtered = df[df["model"].isin(MODELS) & df["dataset_name"].isin(DATASETS)]

    logger.info("Getting top 10 WER samples for each model on each dataset...")
    top_samples = get_top_n_wer_samples(
        df_filtered,
        top_n=10,
        dataset_names=DATASETS,
        models=MODELS,
    )
    
    for dataset_name in DATASETS:
        logger.info(f"Loading dataset: {dataset_name}...")
        if dataset_name == "coral-v2":
            dataset = Dataset.load_from_disk("data/huggingface/datasets/test-sets/CoRal-project--coral-v2-read_aloud-test-unfiltered")
        elif dataset_name == "fleurs":
            dataset = Dataset.load_from_disk("data/huggingface/datasets/test-sets/google--fleurs-da_dk-test-unfiltered")
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

            logger.info(f"Saved top WER samples for model {model} on dataset {dataset_name} to {output_path}")

    for dataset in DATASETS:
        for model in MODELS:
            spearman_correlation_plot(
                df_filtered=df_filtered,
                target_metrics=TARGET_METRICS,
                feature_metrics=FEATURE_METRICS,
                model=model,
                dataset=dataset,
                format_dict=FORMAT_DICT,
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

    