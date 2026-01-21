import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import subprocess
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()
# Configuration
random.seed(42)
DATA_PATH = Path(os.getenv("NEMO_DATASET_PATH")) / "LilleLyd"  # ty:ignore[invalid-argument-type]
PROCESSED_DATA_PATH = (
    Path(os.getenv("NEMO_DATASET_PROCESSED_PATH")) / "LilleLyd"  # ty:ignore[invalid-argument-type]
)
MANIFEST_NAME = "manifest.jsonl"
SAMPLE_RATE = 16000
N_FOLDS = 4


def resample_audio(task):
    """Worker function for parallel resampling."""
    orig_path, new_path = task
    new_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(orig_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(new_path),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {orig_path}: {e}")
        return False


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for entry in data:
            if "emotion" in entry:
                entry["emotion_type"] = entry["emotion"]
                del entry["emotion"]
            entry["source_lang"] = "da"
            entry["target_lang"] = "da"

            f.write(json.dumps(entry) + "\n")


def get_demographics(data):
    demo = {}
    for e in data:
        key = (int(e["age"]), e["gender"])
        demo[key] = demo.get(key, 0) + 1
    return demo


def create_stratified_cv_folds(
    entries: list[dict[str, Any]], n_folds: int = 5
) -> list[tuple[list[dict], list[dict]]]:
    """
    Create stratified cross-validation folds based on age/gender groups.

    Ensures:
    1. No participant appears in both train and test sets within a fold
    2. Groups (age, gender) are balanced across folds

    Args:
        entries: List of manifest entries
        n_folds: Number of CV folds to create

    Returns:
        List of (train_entries, test_entries) tuples for each fold
    """
    # Group entries by (age, gender) -> participant_id -> entries
    group_participants: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        key = (int(entry["age"]), entry["gender"])
        participant_id = entry["participant_id"]
        group_participants[key][participant_id].append(entry)

    # For each group, get list of participant IDs and shuffle
    group_participant_lists: dict[tuple, list[str]] = {}
    for group_key, participant_dict in group_participants.items():
        participants = list(participant_dict.keys())
        random.shuffle(participants)
        group_participant_lists[group_key] = participants

    # Assign participants to folds in a round-robin fashion within each group
    # This ensures balanced distribution across folds
    fold_participants: list[set[str]] = [set() for _ in range(n_folds)]

    for group_key, participants in group_participant_lists.items():
        for i, participant_id in enumerate(participants):
            fold_idx = i % n_folds
            fold_participants[fold_idx].add(participant_id)

    # Create participant to entries mapping
    participant_to_entries: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        participant_to_entries[entry["participant_id"]].append(entry)

    # Generate train/test splits for each fold
    folds = []
    for fold_idx in range(n_folds):
        test_participants = fold_participants[fold_idx]
        train_participants = set()
        for other_fold_idx in range(n_folds):
            if other_fold_idx != fold_idx:
                train_participants.update(fold_participants[other_fold_idx])

        # Verify no overlap
        overlap = test_participants & train_participants
        if overlap:
            raise ValueError(f"Fold {fold_idx}: Found {len(overlap)} overlapping participants!")

        # Collect entries for train and test
        train_entries = []
        test_entries = []

        for participant_id in train_participants:
            train_entries.extend(participant_to_entries[participant_id])

        for participant_id in test_participants:
            test_entries.extend(participant_to_entries[participant_id])

        folds.append((train_entries, test_entries))

    return folds


def create_stratified_sentence_cv_folds(
    entries: list[dict[str, Any]],
) -> list[tuple[list[dict], list[dict]]]:
    """
    Create stratified cross-validation folds based on sentence groups.

    Args:
        entries: List of manifest entries

    Returns:
        List of (train_entries, test_entries) tuples for each fold
    """
    sentences = set(entry["text"] for entry in entries)

    folds = []

    # Leave one-out
    for sentence in sentences:
        train_entries = []
        test_entries = []

        for entry in entries:
            if entry["text"] == sentence:
                test_entries.append(entry)
            else:
                train_entries.append(entry)
        folds.append((train_entries, test_entries))

    return folds


def print_fold_statistics(folds: list[tuple[list[dict], list[dict]]]):
    """Print statistics for each fold to verify balance."""
    logger.info("=" * 60)
    logger.info("Cross-Validation Fold Statistics")
    logger.info("=" * 60)

    for fold_idx, (train_entries, test_entries) in enumerate(folds):
        train_participants = set(e["participant_id"] for e in train_entries)
        test_participants = set(e["participant_id"] for e in test_entries)

        # Get group distribution
        train_groups = defaultdict(int)
        test_groups = defaultdict(int)

        for e in train_entries:
            key = (int(e["age"]), e["gender"])
            train_groups[key] += 1

        for e in test_entries:
            key = (int(e["age"]), e["gender"])
            test_groups[key] += 1

        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(
            f"  Train: {len(train_entries)} entries, {len(train_participants)} participants"
        )
        logger.info(f"  Test:  {len(test_entries)} entries, {len(test_participants)} participants")
        logger.info(
            f"  Overlap check: {len(train_participants & test_participants)} (should be 0)"
        )

        logger.info("  Group distribution (age, gender):")
        all_groups = sorted(set(train_groups.keys()) | set(test_groups.keys()))
        for group in all_groups:
            logger.info(f"    {group}: Train={train_groups[group]}, Test={test_groups[group]}")


def process_lillelyd(print_stats: bool = True):
    manifest_path = DATA_PATH / MANIFEST_NAME
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return

    logger.info("Loading and cleaning manifest")
    raw_entries = load_jsonl(manifest_path)

    # Add recording id
    for idx, entry in enumerate(raw_entries):
        entry["id_recording"] = f"lillelyd_{idx:06d}"

    tasks = []
    for entry in raw_entries:
        if "data/" in entry["audio_filepath"]:
            entry["audio_filepath"] = entry["audio_filepath"].split("data/")[-1]

        orig_p = DATA_PATH / entry["audio_filepath"]
        new_p = PROCESSED_DATA_PATH / entry["audio_filepath"]
        entry["samplerate"] = SAMPLE_RATE

        entry["audio_filepath"] = str(new_p.resolve())

        tasks.append((orig_p, new_p))

    # Parallel Audio Processing
    logger.info(f"Resampling {len(tasks)} files using multiprocessing...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(resample_audio, tasks), total=len(tasks)))

    # Save the updated FULL manifest (cleaned paths + new samplerate)
    full_manifest_path = PROCESSED_DATA_PATH / "manifest.jsonl"
    save_jsonl(raw_entries, full_manifest_path)
    logger.info(f"Saved full processed manifest to {full_manifest_path}")

    # Create stratified CV folds balanced by age/gender groups
    logger.info(f"Creating {N_FOLDS}-fold cross-validation splits (stratified by age/gender)")
    age_gender_fold = create_stratified_cv_folds(raw_entries, n_folds=N_FOLDS)
    sentence_folds = create_stratified_sentence_cv_folds(raw_entries)

    # Print statistics if requested
    if print_stats:
        print_fold_statistics(age_gender_fold)

    # Save each fold's train and test manifests
    cv_dir = PROCESSED_DATA_PATH / "cv_folds"
    cv_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_entries, test_entries) in enumerate(age_gender_fold):
        fold_dir = cv_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_manifest_path = fold_dir / "train_manifest.jsonl"
        test_manifest_path = fold_dir / "test_manifest.jsonl"

        save_jsonl(train_entries, train_manifest_path)
        save_jsonl(test_entries, test_manifest_path)

        logger.info(f"Saved fold {fold_idx + 1}: {train_manifest_path}, {test_manifest_path}")

    logger.info(f"All age/gender {N_FOLDS} CV folds saved to {cv_dir}")

    # Save each fold's train and test manifests
    cv_dir = PROCESSED_DATA_PATH / "cv_folds_sentence"
    cv_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_entries, test_entries) in enumerate(sentence_folds):
        fold_dir = cv_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_manifest_path = fold_dir / "train_manifest.jsonl"
        test_manifest_path = fold_dir / "test_manifest.jsonl"

        save_jsonl(train_entries, train_manifest_path)
        save_jsonl(test_entries, test_manifest_path)

        logger.info(f"Saved fold {fold_idx + 1}: {train_manifest_path}, {test_manifest_path}")

    logger.info(f"All sentence {N_FOLDS} CV folds saved to {cv_dir}")


if __name__ == "__main__":
    process_lillelyd()
