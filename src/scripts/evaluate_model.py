"""Evaluate a speech model.

Usage:
    python src/scripts/evaluate_model.py [key=value] [key=value] ...
"""

import json
import logging

from dotenv import load_dotenv
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

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

    results = evaluate(config=config.eval)

    hydra_output_dir = HydraConfig.get().runtime.output_dir

    logger.info(f"Saving results to {hydra_output_dir}/results.json...")

    with open(f"{hydra_output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
