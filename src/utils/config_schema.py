import dataclasses
from enum import Enum
import warnings

from pydantic.dataclasses import dataclass
from pydantic.types import DirectoryPath, PositiveFloat, PositiveInt

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
    chunk_length_s: PositiveFloat
    stride_length_s: PositiveFloat

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
class ConfigSchema:
    """Configuration schema for the evaluation script."""

    dataset: DatasetConfigSchema
    model: ModelConfigSchema
    eval: EvaluationConfigSchema

    enable_wandb: bool = True
