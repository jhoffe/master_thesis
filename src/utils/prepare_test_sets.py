"""
Prepare evaluation data by combining detailed results, computing sentence embeddings,
and calculating average metrics.
"""
from pathlib import Path
import os

from loguru import logger

from datasets import Dataset

import librosa
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

DATASETS = {
    "fleurs": "google--fleurs-da_dk-test-unfiltered",
    "coral": "CoRal-project--coral-v2-read_aloud-test-unfiltered",
}


def extract_pitch_librosa(y,
                          sr,
                          fmin=50,
                          fmax=350,
                          frame_length=2048,
                          hop_length=256):
    """
    Returns:
      f0_hz: numpy array of pitch values in Hz (NaN = unvoiced)
      voiced_ratio: fraction of voiced frames
      mean_pitch_hz: mean F0 over voiced frames
      median_pitch_hz: median F0 over voiced frames
    """

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Pitch tracking with pyin
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Remove unvoiced frames
    f0_voiced = f0[~np.isnan(f0)]

    if len(f0_voiced) == 0:
        return f0, 0.0, float('nan'), float('nan')

    return f0, len(f0_voiced) / len(f0), float(np.mean(f0_voiced)), float(np.median(f0_voiced))


def process_sample(sample, dataset_name):
    if dataset_name == "coral":
        id = sample['id_recording']
    elif dataset_name == "fleurs":
        id = sample['id']
    else:
        id = None

    y, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]

    _, voiced_ratio, mean_pitch_hz, median_pitch_hz = extract_pitch_librosa(y, sr)
    return {
        "id": id,
        "voiced_ratio": voiced_ratio,
        "mean_pitch_hz": mean_pitch_hz,
        "median_pitch_hz": median_pitch_hz,
    }

def prepare_test_sets() -> None:
    """
    Prepare test sets for evaluation by loading datasets and saving them locally.
    """
    for dataset_name, path in DATASETS.items():
        logger.info(f"Loading dataset: {dataset_name}...")
        evaluation_dataset = Dataset.load_from_disk(f"data/huggingface/datasets/test-sets/{path}")
        logger.info(f"Dataset {dataset_name} loaded with {len(evaluation_dataset)} samples.")
        logger.info(f"Processing samples in dataset: {dataset_name} with joblib...")
        n_jobs = int(os.getenv("N_JOBS", os.cpu_count() or 1))
        samples = list(evaluation_dataset)
        processed_samples = Parallel(n_jobs=n_jobs, backend="loky")( 
            delayed(process_sample)(s, dataset_name) for s in tqdm(samples)
        )

        # make into dataframe
        processed_df = Dataset.from_list(processed_samples).to_pandas()
        processed_df.to_csv(f"reports/metrics/{dataset_name}-processed.csv", index=False)


    