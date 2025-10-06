"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

import logging

from dotenv import load_dotenv
import hydra
from hydra.core.config_store import ConfigStore

from utils.config_schema import ConfigSchema
from utils.evaluate import evaluate

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_evaluation")

cs = ConfigStore.instance()
cs.store(name="evaluation_config_schema", node=ConfigSchema)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: ConfigSchema) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    logger.info("Starting evaluation...")

    results_df = evaluate(config=config.eval)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
