"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from loguru import logger

from utils.ignore_warnings import ignore_warnings
from utils.prepare_evaluation import prepare_evaluation_data

cs = ConfigStore.instance()
# cs.store(name="config_schema", node=ConfigSchema)


def main() -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """

    logger.info("Starting processing of results data...")

    prepare_evaluation_data()

    logger.info("Processing complete.")


if __name__ == "__main__":
    ignore_warnings()
    load_dotenv()

    main()
