"""
Prepare evaluation data by combining detailed results, computing sentence embeddings,
and calculating average metrics.
"""

import os
from pathlib import Path

from datasets import Dataset
from joblib import Parallel, delayed
import librosa
from loguru import logger
import numpy as np
import parselmouth
from tqdm import tqdm

from utils.evaluation_utils import (
    save_to_parquet,
)

from utils.manifest_to_hf import manifest_to_hf_dataset

def extract_pitch_praat(
    y: np.ndarray,
    sr: int,
    time_step: float = 0.01,
    pitch_floor: float = 50.0,
    pitch_ceiling: float = 800.0,
):
    """
    Extract pitch using Praat via parselmouth.

    Returns:
      f0_hz: numpy array of pitch values in Hz (NaN = unvoiced)
      voiced_ratio: fraction of voiced frames
      mean_pitch_hz: mean F0 over voiced frames
      median_pitch_hz: median F0 over voiced frames
    """

    # Create a parselmouth Sound object
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # Extract pitch
    pitch = sound.to_pitch(
        time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling
    )

    # Get pitch values
    f0_values = pitch.selected_array["frequency"]

    # Replace 0 Hz with NaN for unvoiced frames
    f0_values[f0_values == 0] = np.nan

    # Calculate voiced ratio
    num_voiced_frames = np.sum(~np.isnan(f0_values))
    total_frames = len(f0_values)
    voiced_ratio = num_voiced_frames / total_frames if total_frames > 0 else 0.0

    # Calculate mean and median pitch over voiced frames
    f0_voiced = f0_values[~np.isnan(f0_values)]
    mean_pitch_hz = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else float("nan")
    median_pitch_hz = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else float("nan")

    return voiced_ratio, mean_pitch_hz, median_pitch_hz


def process_sample(sample, method="praat"):
    id = sample["id_recording"]

    y, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]

    if method == "praat":
        voiced_ratio, mean_pitch_hz, median_pitch_hz = extract_pitch_praat(y, sr)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "id_recording": id,
        "voiced_ratio": voiced_ratio,
        "mean_pitch_hz": mean_pitch_hz,
        "median_pitch_hz": median_pitch_hz,
    }


def enrich_lillelyd() -> None:
    """
    Prepare test sets for evaluation by loading datasets and saving them locally.
    """

    logger.info("Preparing LilleLyd for evaluation...")
    dataset_name = "lillelyd"
    
    logger.info(f"Loading dataset: {dataset_name}...")
    evaluation_dataset = manifest_to_hf_dataset(
        data_dir=Path("data/processed/lillelyd"),
        manifest_path=Path("data/processed/lillelyd/manifest.jsonl"),
    )
    logger.info(f"Dataset {dataset_name} loaded with {len(evaluation_dataset)} samples.")
    
    # compute clip lengths
    clip_lengths = [x["audio"]["array"].shape[0] / 16000 for x in evaluation_dataset]

    # compute words pr sample
    words_per_sample = [len(x["text"].split()) for x in evaluation_dataset]

    # compute rms energy
    rms_energies = [np.sqrt(np.mean(x["audio"]["array"] ** 2)) for x in evaluation_dataset]

    # compute loudness in dB
    loudness_db = [20 * np.log10(rms + 1e-9) for rms in rms_energies]

    # compute word rate (words per second)
    word_rates = [
        words / length if length > 0 else 0.0
        for words, length in zip(words_per_sample, clip_lengths)
    ]

    logger.info(f"Processing samples in dataset: {dataset_name} with joblib...")
    n_jobs = int(os.getenv("N_JOBS", os.cpu_count() or 1))
    samples = list(evaluation_dataset)
    processed_samples = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_sample)(sample) for sample in tqdm(samples)
    )
    # make into dataframe
    processed_df = Dataset.from_list(processed_samples).to_pandas()

    # Making evaluation dataframe
    evaluation_df = evaluation_dataset.to_pandas()

    # merge on id_recording
    logger.info(f"Merging processed data with original dataset for: {dataset_name}...")
    processed_df = processed_df.merge(evaluation_df, on="id_recording", how="left")

    # drop audio column
    if "audio" in processed_df.columns:
        processed_df = processed_df.drop(columns=["audio"])

    # make clip length column
    processed_df["clip_length"] = np.array(clip_lengths)

    # make word rate column
    processed_df["word_rate"] = np.array(word_rates)

    # make word count column
    processed_df["word_count"] = np.array(words_per_sample)

    # make loudness column
    processed_df["loudness"] = np.array(loudness_db)

    # add dataset name column
    processed_df["dataset_name"] = dataset_name

    # Save summary results to be used in evaluation
    logger.info(f"Saving summary results for dataset: {dataset_name}...")
    save_to_parquet(
        processed_df,
        base_path=Path("reports/metrics"),
        file_name=f"{dataset_name}-summary.parquet",
    )
