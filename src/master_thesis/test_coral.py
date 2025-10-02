from typing import Annotated

from datasets import load_dataset
from dotenv import load_dotenv
import evaluate
import torch
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import typer

load_dotenv(override=True)
app = typer.Typer(name="test_coral")


@app.command()
def test_coral(model: Annotated[str, typer.Argument(...)]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    transcriber = pipeline(task="automatic-speech-recognition", model=model, device=device)
    dataset = load_dataset(
        "CoRal-project/coral-v2",
        name="read_aloud",
        split="test",
        num_proc=8,
    )

    print("Loaded dataset")

    # Hugging Face evaluate WER metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    y = []

    pbar = tqdm(
        transcriber(KeyDataset(dataset, "audio"), batch_size=32),  # type: ignore[arg-type]
        total=len(dataset),
    )
    for out in pbar:
        y.append(out["text"])

    wer = wer_metric.compute(
        predictions=y, references=list(map(lambda x: x.lower(), dataset["text"]))
    )

    print(f"WER: {wer}")

    cer = cer_metric.compute(
        predictions=y, references=list(map(lambda x: x.lower(), dataset["text"]))
    )

    print(f"CER: {cer}")


if __name__ == "__main__":
    app()
