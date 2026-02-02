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


def make_plots():
    logger.info("Loading sentence wise evaluation data with cross-validation folds...")
    sentence_df_folds = load_from_parquet(
        Path("reports/metrics/lillelyd_finetune_combined_detailed_results_with_folds.parquet")
    )

    logger.info("Loading stitched sentence wise evaluation data...")
    sentence_df = load_from_parquet(
        Path("reports/metrics/lillelyd_finetune_stitched_detailed_results.parquet")
    )

    logger.info("Loading summary evaluation data...")
    summary_df = load_from_parquet(Path("reports/metrics/lillelyd_finetune_average_metrics.parquet"))

    for family_name, models in MODEL_FAMILIES.items():

        logger.info(f"Filtering evaluation data to {family_name} grid...")
        sentence_df_filtered = filter_eval_grid(
            sentence_df,
            models=models,
            datasets=DATASETS,
            subsets=SUBSETS,
            splits=SPLITS,
        )

        logger.info(f"Filtering summary data to {family_name} grid...")
        summary_df_filtered = filter_eval_grid(
            summary_df,
            models=models,
            datasets=DATASETS,
            subsets=SUBSETS,
            splits=SPLITS,
        )

        logger.info(f"Generating sentence level evaluation plots for {family_name}...")
        make_all_plots(
            sentence_df_filtered,
            summary_data=summary_df_filtered,
            save_dir=Path(f"reports/csr_finetuning_plots/sentence_level_{family_name.lower()}"),
            width=12,
            height=7,
            font_size=14,
            csr=True,
            models=models,
            df_folds=sentence_df_folds,
        )

        # logger.info(f"Generating summary evaluation plots for {family_name}...")
        # make_all_summary_plots(
        #     summary_df_filtered,
        #     models=models,
        #     save_dir=Path(f"reports/csr_finetuning_plots/summary_{family_name.lower()}"),
        #     width=12,
        #     height=7,
        # )

    logger.info("All plots generated.")
