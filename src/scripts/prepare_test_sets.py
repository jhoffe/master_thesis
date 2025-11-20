"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

from dotenv import load_dotenv
from loguru import logger

from utils.ignore_warnings import ignore_warnings
from utils.prepare_test_sets import prepare_test_sets


def main() -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """

    logger.info("Starting processing of results data...")

    prepare_test_sets()

    logger.info("Processing complete.")


if __name__ == "__main__":
    ignore_warnings()
    load_dotenv()

    main()
