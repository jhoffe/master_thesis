import dataclasses
from datetime import timedelta
from enum import Enum
from typing import Iterable, Literal
import warnings

from loguru import Logger
from pydantic.dataclasses import dataclass
from pydantic.types import DirectoryPath, PositiveFloat, PositiveInt
import pytorch_lightning as pl
from pytorch_lightning.profilers import Profiler

warnings.filterwarnings("ignore", category=UserWarning)


class Metric(str, Enum):
    CER = "cer"
    WER = "wer"


@dataclass(frozen=True)
class EvaluationConfigSchema:
    metrics: list[Metric] = dataclasses.field(default_factory=lambda: [Metric.WER, Metric.CER])
    batch_size: PositiveInt = 16
    store_results: bool = True

    num_workers: PositiveInt = 4

    debug: bool = False


@dataclass(frozen=True)
class ModelConfigSchema:
    name: str

    model_id: str
    chunk_length_s: PositiveFloat | None = None
    stride_length_s: PositiveFloat | None = None

    nemo_model: bool = False

    # Evaluation parameters
    no_lm: bool = False  # This is only relevant for Wav2Vec 2.0 models

    task: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class DatasetConfigSchema:
    name: str

    dataset_id: str
    dataset_subset: str
    eval_split_name: str

    cache_dir: DirectoryPath | None = None

    id_column: str | None = None

    text_column: str = "text"
    audio_column: str = "audio"

    characters_to_keep: str = "abcdefghijklmnopqrstuvwxyzæøå0123456789éü"

    # Filtering of the dataset
    filter: bool = False
    min_seconds_per_example: PositiveFloat = 0.5
    max_seconds_per_example: PositiveInt = 10

    # Processing of the dataset
    clean_text: bool = True
    lower_case: bool = True
    sampling_rate: PositiveInt = 16_000


@dataclass(frozen=True)
class TrainerConfigSchema:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: list[int] | str | int = "auto"
    num_nodes: int = 1
    precision: str | int | None = None
    logger: Logger | Iterable[Logger] | bool | None = None
    callbacks: list[pl.Callback] | pl.Callback | None = None
    fast_dev_run: int | bool = False
    max_epochs: int | None = None
    min_epochs: int | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_time: str | timedelta | dict[str, int] | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0.0
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    num_sanity_val_steps: int | None = None
    log_every_n_steps: int | None = None
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: str | None = None
    deterministic: bool | Literal["warn"] | None = None
    benchmark: bool | None = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Profiler | str | None = None
    model_registry: str | None = None


@dataclass(frozen=True)
class ConfigSchema:
    """Configuration schema for the evaluation script."""

    dataset: DatasetConfigSchema
    model: ModelConfigSchema
    eval: EvaluationConfigSchema
    trainer: TrainerConfigSchema

    enable_wandb: bool = True
