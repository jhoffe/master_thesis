"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

import json

from dotenv import load_dotenv
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from utils.config_schema import ConfigSchema
from utils.ignore_warnings import ignore_warnings
from utils.process_results_data import process_results_data

cs = ConfigStore.instance()
#cs.store(name="config_schema", node=ConfigSchema)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: ConfigSchema) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    
    logger.info("Starting processing of results data...")

    metrics = process_results_data(config=config)

    logger.info("Processing complete.")


if __name__ == "__main__":
    ignore_warnings()
    load_dotenv()

    main()
