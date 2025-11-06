"""
Prepare evaluation data by combining detailed results, computing sentence embeddings,
and calculating average metrics.
"""
from pathlib import Path

from loguru import logger

from utils.evaluation_utils import (
    combine_all_detailed_results,
    compute_average_metrics_for_detailed_results,
    compute_sentence_embeddings,
    save_to_parquet,
)

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
    "canary-1b-v2_finetuned_spec-aug",
    "parakeet-tdt-0.6b-v3_finetuned_spec-aug",

]

DATASETS = [
    "coral-v2",
    "fleurs",
]

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
        "model": model,
        "dataset_name": dataset,
        "dataset_subset": SUBSETS[dataset],
        "dataset_split": SPLITS[dataset]
    }
    for model in MODELS
    for dataset in DATASETS
]

SENTENCE_TRANSFORMER_MODEL = "KennethTM/MiniLM-L6-danish-encoder"

def prepare_evaluation_data() -> None:
    """
    Prepare evaluation data by processing results and computing metrics.

    Returns:
        A dictionary with the evaluation metrics.
    """
    base_path = Path("reports/metrics")
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info("Combining detailed results from all evaluations...")
    detailed_results_df = combine_all_detailed_results(eval_combination=EVALUATION_COMBINATIONS)
    logger.info("Saving combined detailed results to parquet...")
    save_to_parquet(
        df=detailed_results_df,
        base_path=base_path,
        file_name="combined_detailed_results.parquet"
    )
    logger.info(f"Saved {len(detailed_results_df)} rows of detailed results.")

    logger.info("Computing sentence embeddings for detailed results...")
    detailed_results_with_embeddings_df = compute_sentence_embeddings(
        df=detailed_results_df,
        model_name=SENTENCE_TRANSFORMER_MODEL,
    )

    logger.info("Saving detailed results with embeddings to parquet...")
    save_to_parquet(
        df=detailed_results_with_embeddings_df,
        base_path=base_path,
        file_name="combined_detailed_results_with_embeddings.parquet"
    )
    logger.info(f"Saved {len(detailed_results_with_embeddings_df)} rows of detailed results with embeddings.")

    logger.info("Computing average metrics for detailed results...")
    average_metrics_df = compute_average_metrics_for_detailed_results(
        df=detailed_results_with_embeddings_df,
        eval_combinations=EVALUATION_COMBINATIONS,
    )

    logger.info("Saving average metrics to parquet...")
    save_to_parquet(
        df=average_metrics_df,
        base_path=base_path,
        file_name="average_metrics.parquet"
    )
    logger.info(f"Saved {len(average_metrics_df)} rows of average metrics.")
    logger.info("Evaluation data preparation complete.")
