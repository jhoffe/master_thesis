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
from utils.evaluate import evaluate
from utils.ignore_warnings import ignore_warnings
from utils.wandb_setup import WandbSetup


cs = ConfigStore.instance()
cs.store(name="config_schema", node=ConfigSchema)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: ConfigSchema) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """

    with WandbSetup(
        config=config,
        job_type="evaluation",
        tags=(config.model.name, config.dataset.name),
    ):
        logger.info("Starting evaluation...")

        results = evaluate(config=config)

        hydra_output_dir = HydraConfig.get().runtime.output_dir

        logger.info(f"Saving results to {hydra_output_dir}/results.json...")

        with open(f"{hydra_output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Evaluation complete.")


if __name__ == "__main__":
    ignore_warnings()
    load_dotenv()

    main()
