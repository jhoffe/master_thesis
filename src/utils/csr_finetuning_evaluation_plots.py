from pathlib import Path

from loguru import logger

from utils.evaluation_utils import (
    filter_eval_grid,
    load_from_parquet,
)
from utils.plot_sentence_metrics import (
    make_all_plots,
)
from utils.plot_summary_metrics import (
    make_all_summary_plots,
)

# =========================
# Configuration
# =========================
MODELS = [
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

SUBSET_MODELS = [
    "canary-finetune_spec-aug_speed-perturbations",
    "canary-finetune_SA_SP_ll_SA",
    #"canary-finetune_SA_SP_ll_SA_PS",
    "parakeet-finetune_spec-aug",
    "parakeet-finetune_SA_ll_SA_SP",
]

PARAKEET_MODELS = [model for model in MODELS if model.startswith("parakeet")]
CANARY_MODELS = [model for model in MODELS if model.startswith("canary")]

MODEL_FAMILIES = {
    "Parakeet": PARAKEET_MODELS,
    "Canary": CANARY_MODELS
}

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

CV_FOLDS = {
    "cv-1",
    "cv-2",
    "cv-3",
    "cv-4",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
    "lillelyd": "test",
}


def make_plots(type: str = "speaker") -> None:
    logger.info(f"Loading sentence wise evaluation data with {type} cross-validation folds...")
    if type not in ["speaker", "sentence"]:
        raise ValueError(f"Invalid type: {type}. Must be 'speaker' or 'sentence'.")
    
    sentence_df_folds = load_from_parquet(
        Path(f"reports/metrics/lillelyd_finetune_{type}_combined_detailed_results_with_folds.parquet")
    )

    logger.info("Loading stitched sentence wise evaluation data...")
    sentence_df = load_from_parquet(
        Path(f"reports/metrics/lillelyd_finetune_{type}_stitched_detailed_results.parquet")
    )

    logger.info("Loading summary evaluation data...")
    summary_df = load_from_parquet(Path(f"reports/metrics/lillelyd_finetune_{type}_average_metrics.parquet"))

    logger.info("Generating evaluation plots for each model family...")
    
    # for family_name, models in MODEL_FAMILIES.items():

    #     logger.info(f"Filtering evaluation data to {family_name} grid...")
    #     sentence_df_filtered = filter_eval_grid(
    #         sentence_df,
    #         models=models,
    #         datasets=DATASETS,
    #         subsets=SUBSETS,
    #         splits=SPLITS,
    #     )

    #     logger.info(f"Filtering summary data to {family_name} grid...")
    #     summary_df_filtered = filter_eval_grid(
    #         summary_df,
    #         models=models,
    #         datasets=DATASETS,
    #         subsets=SUBSETS,
    #         splits=SPLITS,
    #     )

    #     logger.info(f"Generating sentence level evaluation plots for {family_name}...")
    #     make_all_plots(
    #         sentence_df_filtered,
    #         summary_data=summary_df_filtered,
    #         save_dir=Path(f"reports/csr_finetuning_plots_{type}/sentence_level_{family_name.lower()}"),
    #         width=12,
    #         height=7,
    #         csr=True,
    #         models=models,
    #         df_folds=sentence_df_folds,
    #         type=type,
    #     )

    logger.info("Generating evaluation plots for subset models...")
    sentence_df_subset = filter_eval_grid(
        sentence_df,
        models=SUBSET_MODELS,
        datasets=DATASETS,
        subsets=SUBSETS,
        splits=SPLITS,
    )
    summary_df_subset = filter_eval_grid(
        summary_df,
        models=SUBSET_MODELS,
        datasets=DATASETS,
        subsets=SUBSETS,
        splits=SPLITS,
    )
    make_all_plots(
        sentence_df_subset,
        summary_data=summary_df_subset,
        save_dir=Path(f"reports/csr_finetuning_plots_{type}/sentence_level_subset"),
        width=5,
        height=4,
        csr=True,
        models=SUBSET_MODELS,
        df_folds=sentence_df_folds,
        type=type
    )

    logger.info("All plots generated.")
