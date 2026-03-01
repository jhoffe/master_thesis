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
]

DATASETS = ["coral-v2", "fleurs"]

SUBSETS = {
    "coral-v2": "read_aloud",
    "fleurs": "da_dk",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
}


def make_plots():
    logger.info("Loading sentence wise evaluation data...")
    sentence_df = load_from_parquet(
        Path("reports/metrics/combined_detailed_results_with_embeddings.parquet")
    )

    logger.info("Filtering evaluation data to specified grid...")
    sentence_df = filter_eval_grid(
        sentence_df,
        models=MODELS,
        datasets=DATASETS,
        subsets=SUBSETS,
        splits=SPLITS,
    )

    logger.info("Loading summary evaluation data...")
    summary_df = load_from_parquet(Path("reports/metrics/average_metrics.parquet"))

    logger.info("Filtering summary data to specified grid...")
    summary_df = filter_eval_grid(
        summary_df,
        models=MODELS,
        datasets=DATASETS,
        subsets=SUBSETS,
        splits=SPLITS,
    )

    logger.info("Generating sentence level evaluation plots...")
    make_all_plots(
        sentence_df,
        summary_data=summary_df,
        save_dir=Path("reports/plots/sentence_level"),
        width=8,
        height=3,
    )

    logger.info("Loading summary evaluation data...")
    summary_df = load_from_parquet(Path("reports/metrics/average_metrics.parquet"))

    logger.info("Generating summary evaluation plots...")
    make_all_summary_plots(
        summary_df,
        models=MODELS,
        save_dir=Path("reports/plots/summary"),
        width=8,
        height=5,
    )

    logger.info("All plots generated.")
