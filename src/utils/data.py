"""Functions related to the data loading and processing."""

from collections.abc import Callable, Iterable
from functools import partial
import os
from pathlib import Path
import re
from typing import Any
from unicodedata import normalize

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loguru import logger

from utils.config_schema import ConfigSchema

from .project_types import Data
from .utils import NUMERAL_REGEX, convert_iterable_dataset_to_dataset, convert_numeral_to_words

# Dictionary that contains characters to be converted (from the key to the value). Some
# values contain spaces to ensure that they're separated from other characters, and
# superfluous spaces are removed later. Note also that these are converted in the order
# they appear in the dictionary.
DEFAULT_CONVERSION_DICT = {
    "aa": "å",
    "ğ": "g",
    "ñ": "n",
    "ń": "n",
    "è": "e",
    "kg": " kilo ",
    "μg": " mikrogram ",
    "-": " minus ",
    "+": " plus ",
    "μ": " mikro ",
    "§": " paragraf ",
    "%": " procent ",
    "‰": " promille ",
    "ú": "u",
    "ş": "s",
    "ê": "e",
    "ã": "a",
    "ë": "e",
    "ć": "c",
    "ä": "æ",
    "í": "i",
    "š": "s",
    "î": "i",
    "ě": "e",
    "ð": "d",
    "á": "a",
    "ó": "o",
    "þ": "th",
    "ı": "i",
    "ö": "ø",
    "ç": "c",
    "ș": "s",
    "\u0301": " ",  # Empty whitespace symbol
    "\u200b": " ",  # Empty whitespace symbol
}


def load_dataset_for_evaluation(config: ConfigSchema) -> Dataset:
    """Load the evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DatasetDict containing the validation and test datasets.
    """

    dataset_id = config.dataset.dataset_id
    dataset_subset = config.dataset.dataset_subset

    logger.info(
        f"Loading the {config.dataset.eval_split_name!r} split of the subset {dataset_subset} from {dataset_id} dataset..."
    )

    if config.dataset.cache_dir:
        eval_dataset_path = make_path(config, dataset_id, dataset_subset)

        if eval_dataset_path.exists():
            return Dataset.load_from_disk(dataset_path=eval_dataset_path)

    dataset = load_dataset(
        path=dataset_id,
        name=dataset_subset,
        split=config.dataset.eval_split_name,
        token=os.getenv("HF_AUTH_TOKEN", True),
        cache_dir=str(config.dataset.cache_dir) if config.dataset.cache_dir else None,
        streaming=True,
        trust_remote_code=True,
    )
    assert isinstance(dataset, IterableDataset)

    dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset,
        split_name=config.dataset.eval_split_name,
        cache_dir=config.dataset.cache_dir,
    )

    logger.info(f"Dataset converted with {len(dataset):,} samples.")

    assert isinstance(dataset, Dataset)

    # check if dataset if fleurs
    if config.dataset.name == "fleurs":
        ids = [f"rec_{idx}" for idx in range(1, len(dataset) + 1)]
        dataset = dataset.add_column(name="id_recording", column=ids)

    # filter the dataset
    if config.dataset.filter:
        dataset = filter_dataset(
            dataset=dataset,
            audio_column=config.dataset.audio_column,
            min_seconds_per_example=config.dataset.min_seconds_per_example,
            max_seconds_per_example=config.dataset.max_seconds_per_example,
        )
        logger.info(f"Filtered dataset has {len(dataset):,} samples.")

    dataset = dataset.cast_column(
        column=config.dataset.audio_column,
        feature=Audio(sampling_rate=config.dataset.sampling_rate),
    )
    dataset = process_dataset(
        dataset=dataset,
        clean_text=config.dataset.clean_text,
        lower_case=config.dataset.lower_case,
        characters_to_keep=config.dataset.characters_to_keep,
        text_column=config.dataset.text_column,
        audio_column=config.dataset.audio_column,
        remove_input_dataset_columns=False,
        convert_numerals=True,
    )

    if config.dataset.cache_dir:
        dataset.save_to_disk(dataset_path=eval_dataset_path)  # pyright: ignore[reportPossiblyUnboundVariable]

    return dataset


def make_path(config, dataset_id, dataset_subset):
    return (
        Path(config.dataset.cache_dir)
        / "test-sets"
        / (
            dataset_id.replace("/", "--")
            + f"-{dataset_subset}-{config.dataset.eval_split_name}"
            + ("-filtered" if config.dataset.filter else "-unfiltered")
        )
    )


def filter_dataset(
    dataset: Data,
    audio_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
    num_proc: int | None = None,
) -> Data:
    """Filter the dataset.

    Note that this removes samples from the dataset.

    Args:
        dataset:
            The dataset to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        num_proc (optional):
            The number of processes to use for filtering the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.

    Returns:
        The filtered dataset.
    """

    filter_fn = partial(
        filter_example,
        audio_column=audio_column,
        min_seconds_per_example=min_seconds_per_example,
        max_seconds_per_example=max_seconds_per_example,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        filtered = dataset.filter(function=filter_fn, num_proc=num_proc, desc="Filtering dataset")
    else:
        filtered = dataset.filter(function=filter_fn)

    # Add info back in the filtered dataset, as it gets removed after calling `filter`
    if isinstance(dataset, Dataset | IterableDataset):
        filtered.info.features = dataset.info.features
    else:
        for split_name in dataset.keys():
            dataset[split_name].info.features = filtered[split_name].info.features

    return filtered


def filter_example(
    sample: dict[str, Any],
    audio_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
) -> bool:
    """Filter samples based on the validation status.

    Args:
        sample:
            The sample to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can

    Returns:
        Whether the sample should be kept.
    """
    # Filtering based on audio
    audio = sample[audio_column]
    if audio["array"].shape[0] <= audio["sampling_rate"] * min_seconds_per_example:
        return False
    if audio["array"].shape[0] >= audio["sampling_rate"] * max_seconds_per_example:
        return False

    return True


def process_dataset(
    dataset: Data,
    clean_text: bool,
    lower_case: bool,
    characters_to_keep: Iterable[str] | None,
    text_column: str,
    remove_input_dataset_columns: bool,
    audio_column: str | None,
    convert_numerals: bool,
    num_proc: int | None = None,
    processor: Callable | None = None,
) -> Data:
    """Process the dataset.

    Note that this does not remove any samples from the dataset.

    Args:
        dataset:
            The dataset to be cleaned.
        clean_text:
            Whether to clean the text.
        lower_case:
            Whether to make the text lower case. Only relevant if `clean_text` is True.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept. Only relevant if `clean_text` is True.
        text_column:
            The name of the column containing the text.
        remove_input_dataset_columns:
            Whether to remove all input dataset columns from the output dataset.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        convert_numerals:
            Whether to convert numerals to words.
        num_proc (optional):
            The number of processes to use for processing the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The cleaned dataset.
    """
    if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
        column_names = dataset.column_names

    elif isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        column_names = dataset["train"].column_names

    map_fn = partial(
        process_example,
        characters_to_keep=characters_to_keep,
        conversion_dict=DEFAULT_CONVERSION_DICT,
        text_column=text_column,
        audio_column=audio_column,
        clean_text=clean_text,
        lower_case=lower_case,
        convert_numerals=convert_numerals,
        processor=processor,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        mapped = dataset.map(
            function=map_fn,
            num_proc=num_proc,
            desc="Processing dataset",
            remove_columns=column_names if remove_input_dataset_columns else None,
        )
    else:
        mapped = dataset.map(function=map_fn, remove_columns=column_names)

    return mapped


def process_example(
    example: dict,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str],
    text_column: str,
    audio_column: str | None,
    clean_text: bool,
    lower_case: bool,
    convert_numerals: bool,
    processor: Callable | None,
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        conversion_dict:
            A dictionary of characters to be converted.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        clean_text:
            Whether to clean the text.
        lower_case:
            Whether to make the text lower case.
        convert_numerals:
            Whether to convert numerals to words.
        processor:
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Requires `audio_column` to be specified.

    Returns:
        The cleaned example.
    """
    doc = example[text_column]

    doc = process_text_example(
        text=doc,
        characters_to_keep=characters_to_keep,
        conversion_dict=conversion_dict,
        clean_text=clean_text,
        lower_case=lower_case,
        convert_numerals=convert_numerals,
    )

    # Re-assign the cleaned transcription
    example[text_column] = doc

    if processor is None:
        return example

    # Prepare audio
    audio = example[audio_column]
    sampling_rate = audio["sampling_rate"]
    processed = processor(audio["array"], sampling_rate=sampling_rate)
    if "input_values" in processed:
        example["input_values"] = processed.input_values[0]
        example["num_seconds"] = len(example["input_values"]) / sampling_rate
    if "input_features" in processed:
        example["input_features"] = processed.input_features[0]
        example["num_seconds"] = len(example["input_features"]) / sampling_rate

    # Prepare transcriptions
    example["labels"] = processor(text=example[text_column], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example


def process_text_example(
    text: str,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str] = DEFAULT_CONVERSION_DICT,
    clean_text: bool = True,
    lower_case: bool = True,
    convert_numerals: bool = True,
):
    if convert_numerals and re.search(pattern=NUMERAL_REGEX, string=text):
        text = "".join(
            convert_numeral_to_words(numeral=maybe_numeral)
            for maybe_numeral in re.split(pattern=NUMERAL_REGEX, string=text)
            if maybe_numeral is not None
        )

    if lower_case:
        text = text.lower()

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    if clean_text:
        text = normalize("NFKC", text)

        for key, value in conversion_dict.items():
            text = text.replace(key, value)

        # Remove all non-standard characters
        if characters_to_keep is not None:
            characters_to_keep = "".join(char for char in characters_to_keep)
            if lower_case:
                characters_to_keep = characters_to_keep.lower()
            else:
                characters_to_keep = characters_to_keep.upper() + characters_to_keep.lower()
            non_standard_characters_regex = re.compile(
                f"[^{re.escape(characters_to_keep + ' |')}]"
            )
            text = re.sub(non_standard_characters_regex, " ", text.strip())

        # Replace superfluous spaces
        text = re.sub(r" +", " ", text)

        # Strip each newline
        text = "\n".join([line.strip() for line in text.split("\n")]).strip("\n")
    return text
