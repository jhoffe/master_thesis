"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

from dotenv import load_dotenv
import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger

from utils.config_schema import ConfigSchema
from utils.data import load_dataset_for_evaluation
from utils.ignore_warnings import ignore_warnings

cs = ConfigStore.instance()
cs.store(name="config_schema", node=ConfigSchema)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: ConfigSchema) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """

    logger.info("Loading dataset...")
    load_dataset_for_evaluation(config=config)
    logger.info("Dataset loaded.")


if __name__ == "__main__":
    ignore_warnings()
    load_dotenv()

    main()
