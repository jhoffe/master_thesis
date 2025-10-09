from enum import Enum
from pydantic.dataclasses import dataclass
from pydantic.types import DirectoryPath, PositiveFloat, PositiveInt
import dataclasses


class Metric(str, Enum):
    CER = "cer"
    WER = "wer"


@dataclass
class EvaluationConfigSchema:
    name: str

    model_id: str

    text_column: str = "text"
    audio_column: str = "audio"

    characters_to_keep: str = "abcdefghijklmnopqrstuvwxyzæøå0123456789éü"

    # Evaluation parameters
    no_lm: bool = False  # This is only relevant for Wav2Vec 2.0 models
    metrics: list[Metric] = dataclasses.field(default_factory=lambda: [Metric.WER, Metric.CER])
    batch_size: PositiveInt = 16
    store_results: bool = True

    task: str | None = None
    language: str | None = None

    debug: bool = False


@dataclass
class DatasetConfigSchema:
    name: str

    dataset_id: str
    dataset_subset: str
    eval_split_name: str

    cache_dir: DirectoryPath | None = None

    id_column: str | None = None

    # Filtering of the dataset
    filter: bool = False
    min_seconds_per_example: PositiveFloat = 0.5
    max_seconds_per_example: PositiveInt = 10

    # Processing of the dataset
    clean_text: bool = True
    lower_case: bool = True
    sampling_rate: PositiveInt = 16_000


@dataclass
class ConfigSchema:
    dataset: DatasetConfigSchema
    eval: EvaluationConfigSchema
