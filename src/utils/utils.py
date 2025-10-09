"""General utility functions."""

import collections.abc as c
import multiprocessing as mp
from pathlib import Path
import re

from datasets import (
    Dataset,
    IterableDataset,
    NamedSplit,
    disable_progress_bar,
    enable_progress_bar,
)
from loguru import logger
from tqdm.auto import tqdm

NUMERAL_REGEX = re.compile(r"\b(0|[1-9]\d{0,2}(?:(?:\.\d{3})*|\d*)(?:,\d+)?)\b")


def convert_numeral_to_words(numeral: str, inside_larger_numeral: bool = False) -> str:
    """Convert numerals to words.

    Args:
        numeral:
            The numeral to convert.
        inside_larger_numeral (optional):
            Whether the numeral is inside a larger numeral. For instance, if `numeral`
            is 10, but is part of the larger numeral 1,010, then this should be `True`.

    Returns:
        The text with numerals converted to words.
    """
    if re.fullmatch(pattern=NUMERAL_REGEX, string=numeral) is None:
        return numeral

    numeral = numeral.replace(".", "")

    if "," in numeral:
        assert numeral.count(",") == 1, f"Too many commas in {numeral!r}"
        major, minor = numeral.split(",")
        major = convert_numeral_to_words(numeral=major)
        minor = " ".join(convert_numeral_to_words(numeral=char) for char in minor)
        return f"{major} komma {minor.replace('en', 'et')}"

    match len(numeral):
        case 1:
            mapping = {
                "0": "nul",
                "1": "en",
                "2": "to",
                "3": "tre",
                "4": "fire",
                "5": "fem",
                "6": "seks",
                "7": "syv",
                "8": "otte",
                "9": "ni",
            }
            result = mapping[numeral]

        case 2:
            mapping = {
                "10": "ti",
                "11": "elleve",
                "12": "tolv",
                "13": "tretten",
                "14": "fjorten",
                "15": "femten",
                "16": "seksten",
                "17": "sytten",
                "18": "atten",
                "19": "nitten",
                "20": "tyve",
                "30": "tredive",
                "40": "fyrre",
                "50": "halvtreds",
                "60": "tres",
                "70": "halvfjerds",
                "80": "firs",
                "90": "halvfems",
            }
            if numeral in mapping:
                return mapping[numeral]
            minor = convert_numeral_to_words(numeral=numeral[1], inside_larger_numeral=True)
            major = convert_numeral_to_words(numeral=numeral[0] + "0", inside_larger_numeral=True)
            result = f"{minor}og{major}"

        case 3:
            mapping = {"100": "hundrede"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "hundrede"
            if minor:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 4:
            mapping = {"1000": "tusind"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}".strip()

        case 5:
            major = convert_numeral_to_words(numeral=numeral[:2], inside_larger_numeral=True)
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 6:
            major = convert_numeral_to_words(numeral=numeral[:3], inside_larger_numeral=True)
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 7:
            major = convert_numeral_to_words(numeral=numeral[0], inside_larger_numeral=True)
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "million" if int(numeral[0]) == 1 else "millioner"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 8:
            major = convert_numeral_to_words(numeral=numeral[:2], inside_larger_numeral=True)
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 9:
            major = convert_numeral_to_words(numeral=numeral[:3], inside_larger_numeral=True)
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case _:
            logger.warning(
                f"Cannot convert numerals greater than 999,999,999 to words. Received {numeral!r}"
            )
            return numeral

    return re.sub(r" +", " ", result).strip()


def convert_iterable_dataset_to_dataset(
    iterable_dataset: IterableDataset,
    split_name: str = "train",
    dataset_id: str | None = None,
    cache_dir: Path | None = None,
) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Args:
        iterable_dataset:
            The IterableDataset to convert.
        split_name (optional):
            The name of the split. Defaults to "train".
        dataset_id (optional):
            The ID of the dataset, which is used to store and re-load the dataset. If
            None then the dataset is not stored. Defaults to None.
        cache_dir (optional):
            The directory to store the dataset. If None then the default cache
            `~/.cache/huggingface/datasets` is used. Defaults to None.

    Returns:
        The converted Dataset.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"

    if dataset_id is not None:
        dataset_dir = Path(cache_dir) / dataset_id
        if dataset_dir.exists():
            return Dataset.load_from_disk(str(dataset_dir))

    splits_info = iterable_dataset.info.splits
    num_examples = None if splits_info is None else splits_info[split_name].num_examples

    def gen_from_iterable_dataset() -> c.Generator[dict, None, None]:
        yield from tqdm(
            iterable=iterable_dataset,
            total=num_examples,
            desc="Converting iterable dataset to regular dataset",
        )

    with no_datasets_progress_bars():
        dataset = Dataset.from_generator(
            generator=gen_from_iterable_dataset,
            features=iterable_dataset.features,
            split=NamedSplit(name=split_name),
            num_proc=mp.cpu_count(),
        )
    assert isinstance(dataset, Dataset)

    if dataset_id is not None:
        dataset_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(dataset_dir))

    return dataset


class no_datasets_progress_bars:
    """Context manager that disables the `datasets` progress bars."""

    def __enter__(self) -> None:
        """Disable the progress bar."""
        disable_progress_bar()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        """Re-enable the progress bar."""
        enable_progress_bar()
