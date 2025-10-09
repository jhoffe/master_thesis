import os
from typing import Sequence

import wandb

from utils.config_schema import ConfigSchema


class WandbSetup:
    def __init__(
        self,
        config: ConfigSchema,
        job_type: str,
        tags: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.job_type = job_type
        self.tags = tags

    def __enter__(self):
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            job_type=self.job_type,
            config=dict(self.config),  # type: ignore
            mode="disabled" if not self.config.enable_wandb else "online",
            tags=self.tags,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()
