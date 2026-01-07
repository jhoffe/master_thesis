import json
from pathlib import Path

from datasets import Audio, Dataset


class ManifestHFDataset:
    def __init__(self, data_dir: Path, manifest_path: Path):
        self.data_dir = data_dir
        self.manifest_path = manifest_path

        self.data = list(self._load_manifest())

    def _load_manifest(self):
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                entry["audio"] = str((self.data_dir / entry["audio_filepath"]).resolve())

                yield entry

    def to_hf_dataset(self) -> Dataset:
        hf_dataset = Dataset.from_list(self.data)

        # Set the audio column to be of type Audio
        hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))

        return hf_dataset


def manifest_to_hf_dataset(data_dir: Path, manifest_path: Path) -> Dataset:
    """Convert a manifest file to a Hugging Face Dataset.

    Args:
        manifest_path:
            Path to the manifest file.

    Returns:
        A Hugging Face Dataset.
    """
    manifest_dataset = ManifestHFDataset(data_dir, manifest_path)
    hf_dataset = manifest_dataset.to_hf_dataset()
    return hf_dataset
