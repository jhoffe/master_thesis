from utils.dataset_descriptive_plots import (
    make_coral_plots,
    make_fleurs_plots,
)

from loguru import logger

DATASETS = {
    "fleurs": "google--fleurs-da_dk-test-unfiltered",
    "coral-v2": "CoRal-project--coral-v2-read_aloud-test-unfiltered",
}


METRICS = [
    "clip_length",
    "mean_pitch_hz",
    "median_pitch_hz",
    "voiced_ratio",
]

if __name__ == "__main__":
    logger.info("Generating dataset descriptive plots for CoRal-v2...")
    make_coral_plots()
    logger.info("Generating dataset descriptive plots for Fleurs...")
    make_fleurs_plots()