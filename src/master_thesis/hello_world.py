from typing import Annotated, cast

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
import evaluate
import torch
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import typer


print("Hello world")
