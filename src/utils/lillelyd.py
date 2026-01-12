import json
import subprocess
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from tqdm import tqdm
from loguru import logger

# Configuration
random.seed(42)
DATA_PATH = Path("data/raw/LilleLyd")
PROCESSED_DATA_PATH = Path("data/processed/LilleLyd")
MANIFEST_NAME = "manifest.jsonl"
SAMPLE_RATE = 16000
TEST_SIZE = 0.2

def resample_audio(task):
    """Worker function for parallel resampling."""
    orig_path, new_path = task
    new_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg", "-i", str(orig_path),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(new_path), "-y",
            "-hide_banner", "-loglevel", "error"
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
            f.write(json.dumps(entry) + "\n")

def get_demographics(data):
    demo = {}
    for e in data:
        key = (int(e["age"]), e["gender"])
        demo[key] = demo.get(key, 0) + 1
    return demo

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
        tasks.append((orig_p, new_p))

    # Parallel Audio Processing
    logger.info(f"Resampling {len(tasks)} files using multiprocessing...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(resample_audio, tasks), total=len(tasks)))

    # Save the updated FULL manifest (cleaned paths + new samplerate)
    full_manifest_path = PROCESSED_DATA_PATH / "manifest.jsonl"
    save_jsonl(raw_entries, full_manifest_path)
    logger.info(f"Saved full processed manifest to {full_manifest_path}")

    # Stratified Split by Age and Gender
    logger.info("Splitting data by participant_id (stratified by age/gender)")
    groups = {}
    for entry in raw_entries:
        key = (int(entry["age"]), entry["gender"])
        groups.setdefault(key, []).append(entry)

    train_data, test_data = [], []

    # Sort the group keys so we process age/gender groups in a stable order
    sorted_group_keys = sorted(groups.keys())

    for key in sorted_group_keys:
        entries = groups[key]
        
        # Sort the unique participant IDs. 
        # random.shuffle(list(set(x))) is non-deterministic because sets have no order.
        unique_participants = sorted(list(set(e['participant_id'] for e in entries)))
        
        random.shuffle(unique_participants)
        
        split_idx = ceil(len(unique_participants) * TEST_SIZE)
        
        # Edge case: Ensure at least one speaker stays in train if possible
        if split_idx >= len(unique_participants) and len(unique_participants) > 1:
            split_idx = len(unique_participants) - 1
            
        test_ids = set(unique_participants[:split_idx])
        
        for e in entries:
            if e['participant_id'] in test_ids:
                test_data.append(e)
            else:
                train_data.append(e)

    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Save results
    save_jsonl(train_data, PROCESSED_DATA_PATH / "manifest_train.jsonl")
    save_jsonl(test_data, PROCESSED_DATA_PATH / "manifest_test.jsonl")
    
    logger.success(f"Done! Train: {len(train_data)} utts, Test: {len(test_data)} utts")

    # Sanity checks
    train_ids = set(e['participant_id'] for e in train_data)
    test_ids = set(e['participant_id'] for e in test_data)
    assert train_ids.isdisjoint(test_ids), "Train and Test sets have overlapping participant_ids!"
    
    logger.info(f"Train participant_ids: {sorted(list(train_ids))}")
    logger.info(f"Test participant_ids: {sorted(list(test_ids))}")

    if print_stats:
        logger.info(f"Train demographics: {get_demographics(train_data)}")
        logger.info(f"Test demographics: {get_demographics(test_data)}")

if __name__ == "__main__":
    process_lillelyd()