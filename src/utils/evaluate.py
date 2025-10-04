"""Evaluation of ASR models."""

import itertools as it
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import load_dataset_for_evaluation
from evaluate.loading import load as load_metric

load_dotenv()


logger = logging.getLogger(__package__)

def evaluate(config: DictConfig) -> pd.DataFrame:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    assert config.model_id is not None, (
        "`model_id` must be set to perform an evaluation!"
    )

    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config)

    # Only take the first 25 samples for quick testing
    dataset = dataset.select(range(100))

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(model_id=config.model_id, no_lm=config.no_lm)

    logger.info("Computing the scores...")
    preds, labels, all_scores = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=config.metrics,
        characters_to_keep=config.characters_to_keep,
        text_column=config.text_column,
        audio_column=config.audio_column,
        batch_size=config.batch_size,
    )

    # Hugging Face evaluate WER metric
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    wer = wer_metric.compute(
        predictions=preds, references=labels)
    print(f"WER: {wer}")

    cer = cer_metric.compute(
        predictions=preds, references=labels)
    print(f"CER: {cer}")



#    logger.info(
#        "Converting the dataset to a dataframe computing the scores for each "
#        "metadata category..."
#    )
#    df = convert_evaluation_dataset_to_df(
#        dataset=dataset, sub_dialect_to_dialect_mapping=config.sub_dialect_to_dialect
#   )
#    for metric_name in config.metrics:
#        df[metric_name] = all_scores[metric_name]
#    score_df = get_score_df(
#        df=df,
#        categories=["age_group", "gender", "dialect"],
#        metric_names=config.metrics,
#    )
#    return score_df

def load_asr_pipeline(model_id: str, no_lm: bool) -> AutomaticSpeechRecognitionPipeline:
    """Load the ASR pipeline.

    Args:
        model_id:
            The model ID to load.
        no_lm:
            Whether to load the ASR pipeline without a language model. Only applicable
            to Wav2Vec 2.0 models.

    Returns:
        The ASR pipeline.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if no_lm:
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        transcriber = pipeline(
            task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                dtype=torch.float16 if device.type != "cpu" else torch.float32,
            )
    else:
        transcriber = pipeline(
            task="automatic-speech-recognition", model=model_id, device=device
        )

    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber