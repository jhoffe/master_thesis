import json
import os
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from math import ceil
import random

# set seed for reproducibility
random.seed(42)

DATA_PATH = Path("../data/raw/LilleLyd")
MANIFEST_PATH = DATA_PATH / "manifest.jsonl"
PROCESSED_DATA_PATH = Path("../data/processed/LilleLyd")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def load_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":

    DATA_PATH = Path("../data/raw/LilleLyd")
    MANIFEST_PATH = DATA_PATH / "manifest.jsonl"
    
    logger.info(f"Loading manifest from {MANIFEST_PATH}")
    manifest_lines = list(load_manifest(MANIFEST_PATH))

    logger.info("Modifying audio file paths in manifest")
    for entry in manifest_lines:
        entry["audio_filepath"] = entry["audio_filepath"][len("/Users/joachimschroderandersson/Desktop/exp/data/"):]

    PROCESSED_DATA_PATH = Path("../data/processed/LilleLyd")
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    logger.info("Resampling audio files to 16kHz mono")
    for entry in tqdm(manifest_lines):
        original_path = DATA_PATH / entry["audio_filepath"]
        new_path = PROCESSED_DATA_PATH / entry["audio_filepath"]
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resample using ffmpeg
        os.system(f"ffmpeg -i '{original_path}' -ar 16000 -ac 1 '{new_path}' -y")

    logger.info("Writing new manifest file for processed data")
    new_manifest = PROCESSED_DATA_PATH / "manifest.jsonl"
    with new_manifest.open("w", encoding="utf-8") as f:
        for entry in manifest_lines:
            entry["audio_filepath"] = entry["audio_filepath"]
            entry["samplerate"] = 16000
            f.write(json.dumps(entry) + "\n")

    
    TEST_SIZE = 0.2
    # Split into groups for same age and same gender
    groups: dict[tuple[int, str], list] = {}

    logger.info("Grouping entries by age and gender")
    for entry in manifest_lines:
        age = int(entry["age"])
        gender = entry["gender"]
        key = (age, gender)
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # Shuffle entries within each group    
    for entries in groups.values():
        random.shuffle(entries)

    # Print group sizes
    for key, entries in groups.items():
        num_speakers = len(set(e['participant_id'] for e in entries))

        logger.info(f"Age: {key[0]}, Gender: {key[1]}, Count: {num_speakers} speakers, {len(entries)} utterances")
        logger.info(f"Test speakers: {ceil(num_speakers * TEST_SIZE)}")

    test_manifest = PROCESSED_DATA_PATH / "manifest_test.jsonl"
    train_manifest = PROCESSED_DATA_PATH / "manifest_train.jsonl"

    logger.info("Splitting data into train and test sets")
    with test_manifest.open("w", encoding="utf-8") as f_test, train_manifest.open("w", encoding="utf-8") as f_train:
        for key, entries in groups.items():
            participant_ids = list(set(e['participant_id'] for e in entries))
            random.shuffle(participant_ids)
            num_test_speakers = ceil(len(participant_ids) * TEST_SIZE)
            test_speakers = set(participant_ids[:num_test_speakers])
            
            for entry in entries:
                if entry['participant_id'] in test_speakers:
                    f_test.write(json.dumps(entry) + "\n")
                else:
                    f_train.write(json.dumps(entry) + "\n")
    