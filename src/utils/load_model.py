import os

from loguru import logger
import wandb


def load_model_path(model_path: str) -> str:
    if model_path.startswith("artifact://"):
        logger.info(f"Restoring model from WandB artifact: {model_path}...")

        artifact_full_path = model_path.split("://")[1]
        artifact, artifact_file = artifact_full_path.split("/")

        logger.info(f"Downloading WandB artifact: {artifact} (file: {artifact_file})...")

        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT")

        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{artifact}")
        artifact_dir = artifact.download()

        return os.path.join(artifact_dir, artifact_file)

    return model_path
