import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

scandi_wiki = load_dataset(
    "alexandrainst/scandi-wiki",
    "da",
    split="train",
    trust_remote_code=True,
)
scandi_reddit = load_dataset(
    "alexandrainst/scandi-reddit",
    "da",
    split="train",
    trust_remote_code=True,
)

os.makedirs(os.path.join(os.environ["NEMO_DATASET_PROCESSED_PATH"], "lm"), exist_ok=True)

with open(
    os.path.join(os.environ["NEMO_DATASET_PROCESSED_PATH"], "lm/lm_training.jsonl"),
    "w",
    encoding="utf-8",
) as f:
    for item in tqdm(scandi_wiki, desc="Writing Scandi-Wiki data"):
        f.write(json.dumps({"text": item["text"], "lang": "da"}) + "\n")

    for item in tqdm(scandi_reddit, desc="Writing Scandi-Reddit data"):
        f.write(json.dumps({"text": item["doc"], "lang": "da"}) + "\n")
