"""Evaluation of ASR models."""

import uuid

from dotenv import load_dotenv
from evaluate.loading import load as load_metric
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import pandas as pd
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)
import wandb
from tqdm.auto import tqdm

from utils.config_schema import ConfigSchema, ModelConfigSchema

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import load_dataset_for_evaluation

load_dotenv()


def evaluate(config: ConfigSchema) -> dict[str, float]:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config)

    if config.eval.debug:
        logger.info("Debug mode is on, using only 5 examples from the dataset...")
        dataset = dataset.select(range(64))

    logger.info(f"Loading the {config.model.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(config.model)

    logger.info("Computing the scores...")
    preds, labels, results, rtf, rtfx = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=config.eval.metrics,  # pyright: ignore[reportArgumentType]
        characters_to_keep=config.dataset.characters_to_keep,
        text_column=config.dataset.text_column,
        audio_column=config.dataset.audio_column,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        target_lang=config.model.language,
        id_column=config.dataset.id_column,
        sampling_rate=config.dataset.sampling_rate,
    )

    # Hugging Face evaluate WER metric
    logger.info("Computing WER and CER...")
    wer_metric = load_metric("wer", experiment_id=uuid.uuid4().hex)
    cer_metric = load_metric("cer", experiment_id=uuid.uuid4().hex)

    wer = wer_metric.compute(predictions=preds, references=labels)
    cer = cer_metric.compute(predictions=preds, references=labels)

    logger.info(f"WER: {wer:.4f}, CER: {cer:.4f}, RTF: {rtf:.4f}, RTFx: {rtfx:.4f}")
    wandb.log({"wer": wer, "cer": cer, "rtf": rtf, "rtfx": rtfx})

    wandb.log(
        {
            "detailed_results": wandb.Table(dataframe=pd.DataFrame(results)),
        },
    )

    if config.eval.store_results:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        results_df = pd.DataFrame(results)
        results_df.to_parquet(f"{hydra_output_dir}/detailed_results.parquet", index=False)
        logger.info(f"Saved detailed results to {hydra_output_dir}/detailed_results.parquet")

    return {"wer": wer, "cer": cer, "rtf": rtf, "rtfx": rtfx}


def load_asr_pipeline(config: ModelConfigSchema) -> AutomaticSpeechRecognitionPipeline:
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

    if config.no_lm:
        model = Wav2Vec2ForCTC.from_pretrained(config.model_id)
        processor = Wav2Vec2Processor.from_pretrained(config.model_id)
        arguments = {
            "task": "automatic-speech-recognition",
            "model": model,
            "tokenizer": processor.tokenizer,  # type: ignore
            "feature_extractor": processor.feature_extractor,  # type: ignore
            "device": device,
            "dtype": torch.float16 if device.type != "cpu" else torch.float32,
            "chunk_length_s": config.chunk_length_s if config.chunk_length_s is not None else None,
            "stride_length_s": config.stride_length_s if config.stride_length_s is not None else None,
        }

        transcriber = pipeline(**arguments)
    else:
        arguments = {
            "task": "automatic-speech-recognition",
            "model": config.model_id,
            "device": device,
            "chunk_length_s": config.chunk_length_s,
            "stride_length_s": config.stride_length_s,
        }

        transcriber = pipeline(**arguments)

    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber
