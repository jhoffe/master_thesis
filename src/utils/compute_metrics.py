"""Function used to compute metrics during ASR training of Wav2Vec 2.0 models."""
import uuid
import time

from collections.abc import Iterable

from datasets import Dataset
from evaluate.loading import load as load_metric
from tqdm.auto import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
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
    num_workers: int,
    id_column: str | None = None,
    sampling_rate: int | None = None,
    target_lang: str | None = None,
) -> tuple[list[str], list[str], list]:
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
    metrics = {metric_name: load_metric(metric_name, experiment_id=uuid.uuid4().hex) for metric_name in metric_names}

    if id_column is not None:
        ids = [str(id_) for id_ in dataset[id_column]]
    else:
        ids = list(range(len(dataset)))

    if sampling_rate is not None:
        clip_lengths = [len(sample["array"]) / sampling_rate for sample in dataset[audio_column]]
    else:
        clip_lengths = [None] * len(dataset)

    all_metrics = []

    start_time = time.time()

    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
    ):
        for idx, out in enumerate(
            transcriber(
                KeyDataset(dataset=dataset, key=audio_column),  # type: ignore[arg-type]
                generate_kwargs=dict(language="danish", task="transcribe")
                if target_lang is None
                else dict(tgt_lang=target_lang),
                batch_size=batch_size,
                num_workers=num_workers,
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
                metric_name: metric.compute(predictions=[prediction], references=[labels[idx]])
                for metric_name, metric in metrics.items()
            }
            assert all(isinstance(score, float) for score in scores.values()), (
                f"Expected the scores to be floats, but found {scores}"
            )

            all_metrics.append(scores)
            predictions.append(prediction)
            pbar.update()

    end_time = time.time()
    duration = end_time - start_time

    # compute RTFx
    if all(clip_length is not None for clip_length in clip_lengths):
        total_audio_length = sum(clip_length for clip_length in clip_lengths if clip_length is not None)
        rtf = duration / total_audio_length
        rtfx = 1 / rtf
    else:
        rtf = None
        rtfx = None

    # gather results
    results = [
        {
            "id": ids[idx],
            "prediction": predictions[idx],
            "label": labels[idx],
            "clip_length": clip_lengths[idx],
            "wer": all_metrics[idx].get("wer"),
            "cer": all_metrics[idx].get("cer"),
        }
        for idx in range(len(dataset))
    ]

    return predictions, labels, results, rtf, rtfx
