"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

import logging
from pathlib import Path
from shutil import rmtree

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from utils.evaluate import evaluate


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_evaluation")


@hydra.main(config_path="../../config", config_name="evaluation_roest", version_base=None)
def main(config: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    logger.info("Starting evaluation...")

    results_df = evaluate(config=config)

    logger.info("Evaluation complete.")


    
if __name__ == "__main__":
    main()