"""Function used to compute metrics during ASR training of Wav2Vec 2.0 models."""

import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import DefaultDict

import numpy as np
from datasets import Dataset
from evaluate.loading import load as load_metric
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Wav2Vec2ProcessorWithLM,
)
from transformers.pipelines.pt_utils import KeyDataset

from .data import DEFAULT_CONVERSION_DICT, process_example

def compute_metrics_of_dataset_using_pipeline(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    characters_to_keep: Iterable[str],
    text_column: str,
    audio_column: str,
    batch_size: int,
) -> tuple[list[str], list[str], dict[str, list[float]]]:
    """Compute the metrics for the dataset using a pipeline.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.
        metric_names:
            The names of the metrics to compute. Needs to be compatible with the name of
            the metric in the `evaluate` library.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, all_scores) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            all_scores:
                A dictionary containing the computed scores for each metric.
    """
    characters_to_keep = "".join(characters_to_keep)

    labels: list[str] = [lbl.strip().lower() for lbl in dataset[text_column]]
    predictions: list[str] = list()
    metrics = {metric_name: load_metric(metric_name) for metric_name in metric_names}
    all_scores: DefaultDict = defaultdict(list)

    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
    ):
        for idx, out in enumerate(
            transcriber(
                KeyDataset(dataset=dataset, key=audio_column),  # type: ignore[arg-type]
                batch_size=batch_size,
                generate_kwargs=dict(language="danish", task="transcribe"),
            )
        ):
            prediction = process_example(
                example=dict(text=out["text"]),
                characters_to_keep=characters_to_keep,
                conversion_dict=DEFAULT_CONVERSION_DICT,
                text_column="text",
                audio_column=None,
                clean_text=True,
                lower_case=True,
                convert_numerals=True,
                processor=None,
            )["text"]

            scores = {
                metric_name: metric.compute(
                    predictions=[prediction], references=[labels[idx]]
                )
                for metric_name, metric in metrics.items()
            }
            assert all(isinstance(score, float) for score in scores.values()), (
                f"Expected the scores to be floats, but found {scores}"
            )

            for metric, score in scores.items():
                all_scores[metric].append(score)
            predictions.append(prediction)
            pbar.update()

    return predictions, labels, all_scores