"""
Prepare evaluation data by combining detailed results, computing sentence embeddings,
and calculating average metrics.
"""

from pathlib import Path

from loguru import logger

from utils.evaluation_csr_utils import (
    combine_all_detailed_results_lillelyd,
    make_stitched_lillelyd_df,
    compute_average_metrics_for_detailed_results,
    save_to_parquet,
    load_from_parquet,
)

SPEAKER_MODELS = [
    "roest-whisper-large-v1",
    "canary-finetune_spec-aug_speed-perturbations",
    "canary-finetune_SA_SP_ll",
    "canary-finetune_SA_SP_ll_SA",
    "canary-finetune_SA_SP_ll_PS",
    "canary-finetune_SA_SP_ll_SP",
    "canary-finetune_SA_SP_ll_SA_PS",
    "canary-finetune_SA_SP_ll_SA_SP",
    "canary-finetune_SA_SP_ll_PS_SP",
    "canary-finetune_SA_SP_ll_SA_PS_SP",
    "parakeet-finetune_spec-aug",
    "parakeet-finetune_SA_ll",
    "parakeet-finetune_SA_ll_SA",
    "parakeet-finetune_SA_ll_PS",
    "parakeet-finetune_SA_ll_SP",
    "parakeet-finetune_SA_ll_SA_PS",
    "parakeet-finetune_SA_ll_SA_SP",
    "parakeet-finetune_SA_ll_PS_SP",
    "parakeet-finetune_SA_ll_SA_PS_SP",
]

SENTENCE_MODELS = [
    "canary-finetune_spec-aug_speed-perturbations",
    "canary-finetune_SA_SP_ll",
    "canary-finetune_SA_SP_ll_SA",
    "canary-finetune_SA_SP_ll_PS",
    "canary-finetune_SA_SP_ll_SP",
    "canary-finetune_SA_SP_ll_SA_PS",
    "canary-finetune_SA_SP_ll_SA_SP",
    "canary-finetune_SA_SP_ll_PS_SP",
    "canary-finetune_SA_SP_ll_SA_PS_SP",
    "parakeet-finetune_spec-aug",
    "parakeet-finetune_SA_ll",
    "parakeet-finetune_SA_ll_SA",
    "parakeet-finetune_SA_ll_PS",
    "parakeet-finetune_SA_ll_SP",
    "parakeet-finetune_SA_ll_SA_PS",
    "parakeet-finetune_SA_ll_SA_SP",
    "parakeet-finetune_SA_ll_PS_SP",
    "parakeet-finetune_SA_ll_SA_PS_SP",
]

DATASETS = [
    "coral-v2",
    "fleurs",
    "lillelyd"
]

SUBSETS = {
    "coral-v2": "read_aloud",
    "fleurs": "da_dk",
    "lillelyd": "full",
}

SPEAKER_CV_FOLDS = {
    "cv-1",
    "cv-2",
    "cv-3",
    "cv-4",
}

SENTENCE_CV_FOLDS = {
    "sentence_cv-1",
    "sentence_cv-2",
    "sentence_cv-3",
    "sentence_cv-4",
    "sentence_cv-5",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
    "lillelyd": "test",
}

def provide_combinations(models: list, folds: set, include_folds: bool = True):
    combinations = []
    if include_folds:
        for m in models:
            for d in DATASETS:
                if "ll" in m:
                    for f in folds:
                        combinations.append(
                            {
                                "model": m,
                                "dataset_name": d,
                                "dataset_subset": SUBSETS[d],
                                "dataset_split": SPLITS[d],
                                "cv_fold": f,
                            }
                        )
                else:
                    if d == "lillelyd":
                        for f in folds:
                            combinations.append(
                                {
                                    "model": m,
                                    "dataset_name": d,
                                    "dataset_subset": SUBSETS[d],
                                    "dataset_split": SPLITS[d],
                                    "cv_fold": f,
                                }
                            )
                    else:
                        combinations.append(
                            {
                                "model": m,
                                "dataset_name": d,
                                "dataset_subset": SUBSETS[d],
                                "dataset_split": SPLITS[d],
                                "cv_fold": None,
                            }
                        )
    else:
        for m in models:
            for d in DATASETS:
                combinations.append(
                    {
                        "model": m,
                        "dataset_name": d,
                        "dataset_subset": SUBSETS[d],
                        "dataset_split": SPLITS[d],
                    }
                )
    return combinations

SPEAKER_EVALUATION_COMBINATIONS_FOLDS = provide_combinations(
    models=SPEAKER_MODELS, 
    folds=SPEAKER_CV_FOLDS, 
    include_folds=True
)
SPEAKER_EVALUATION_COMBINATIONS = provide_combinations(
    models=SPEAKER_MODELS, 
    folds=SPEAKER_CV_FOLDS, 
    include_folds=False
)

SENTENCE_EVALUATION_COMBINATIONS_FOLDS = provide_combinations(
    models=SENTENCE_MODELS, 
    folds=SENTENCE_CV_FOLDS, 
    include_folds=True
)

SENTENCE_EVALUATION_COMBINATIONS = provide_combinations(
    models=SENTENCE_MODELS, 
    folds=SENTENCE_CV_FOLDS, 
    include_folds=False
)

def prepare_evaluation_data(type: str) -> None:
    """
    Prepare evaluation data by processing results and computing metrics.

    Returns:
        A dictionary with the evaluation metrics.
    """
    base_path = Path("reports/metrics")
    base_path.mkdir(parents=True, exist_ok=True)

    if type == "speaker":
        evaluation_combinations_folds = SPEAKER_EVALUATION_COMBINATIONS_FOLDS
        evaluation_combinations = SPEAKER_EVALUATION_COMBINATIONS
    elif type == "sentence":
        evaluation_combinations_folds = SENTENCE_EVALUATION_COMBINATIONS_FOLDS
        evaluation_combinations = SENTENCE_EVALUATION_COMBINATIONS
    else:
        raise ValueError(f"Invalid type: {type}. Must be 'speaker' or 'sentence'.")

    logger.info(f"Preparing {type} evaluation data...")

    logger.info(f"Combining detailed results from all evaluations (total of {len(evaluation_combinations_folds)} combinations)...")
    detailed_results_df = combine_all_detailed_results_lillelyd(
        combinations=evaluation_combinations_folds, 
        base="experiments/evaluate_model"
        )

    logger.info("Saving combined detailed results to parquet...")
    save_to_parquet(
        df=detailed_results_df, 
        base_path=base_path, 
        file_name=f"lillelyd_finetune_{type}_combined_detailed_results_with_folds.parquet"
    )
    logger.info(f"Saved {len(detailed_results_df)} rows of detailed results.")

    logger.info(f"Creating stitched dataframe plotting for {type}...")
    stitched_df = make_stitched_lillelyd_df(combined_df=detailed_results_df)
    
    logger.info(f"Saving stitched detailed results to parquet for {type}...")
    save_to_parquet(
        df=stitched_df, 
        base_path=base_path, 
        file_name=f"lillelyd_finetune_{type}_stitched_detailed_results.parquet"
    )
    logger.info(f"Saved {len(stitched_df)} rows of stitched detailed results.")

    logger.info(f"Loading combined detailed results from parquet for {type}...")
    stitched_df = load_from_parquet(
        Path(f"reports/metrics/lillelyd_finetune_{type}_stitched_detailed_results.parquet")
    )

    logger.info(f"Computing average metrics for stitched detailed results (total of {len(evaluation_combinations)} combinations)...")
    average_metrics_df = compute_average_metrics_for_detailed_results(
        df=stitched_df,
        eval_combinations=evaluation_combinations,
    )

    logger.info("Saving average metrics to parquet...")
    save_to_parquet(
        df=average_metrics_df, 
        base_path=base_path, 
        file_name=f"lillelyd_finetune_{type}_average_metrics.parquet"
    )
    logger.info(f"Saved {len(average_metrics_df)} rows of average metrics.")

    logger.info("Evaluation data preparation complete.")
