"""Evaluation of ASR models."""

import logging

from dotenv import load_dotenv
from evaluate.loading import load as load_metric
from omegaconf import DictConfig
import pandas as pd
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

from utils.config_schema import EvaluationConfigSchema

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import load_dataset_for_evaluation

load_dotenv()


logger = logging.getLogger(__package__)


def evaluate(config: EvaluationConfigSchema) -> dict[str, float]:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    assert config.model_id is not None, "`model_id` must be set to perform an evaluation!"

    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config)

    if config.debug:
        logger.info("Debug mode is on, using only 5 examples from the dataset...")
        dataset = dataset.select(range(5))

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

    wer = wer_metric.compute(predictions=preds, references=labels)
    cer = cer_metric.compute(predictions=preds, references=labels)

    return {"wer": wer, "cer": cer}


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
        transcriber = pipeline(task="automatic-speech-recognition", model=model_id, device=device)

    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber
