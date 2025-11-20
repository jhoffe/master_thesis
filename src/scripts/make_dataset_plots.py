from loguru import logger

from utils.dataset_descriptive_plots import (
    make_coral_plots,
    make_fleurs_plots,
)

if __name__ == "__main__":
    logger.info("Generating dataset descriptive plots for CoRal-v2...")
    make_coral_plots()
    logger.info("Generating dataset descriptive plots for Fleurs...")
    make_fleurs_plots()
