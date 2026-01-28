import json
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a manifest file and return its contents as a list of dictionaries.

    Args:
        manifest_path (str): Path to the manifest file.
    Returns:
        list: A list of dictionaries representing the manifest entries.
    """

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = [json.loads(line.strip()) for line in f if line.strip()]
    return manifest


def process_sample(sample: dict, output_path: Path, perturbation: float):
    """Process a single sample from the manifest.

    Args:
        sample (dict): A dictionary representing a single sample from the manifest.
    Returns:
        dict: The processed sample.
    """
    audio_path = Path(sample["audio_filepath"])

    full_output_path = output_path / audio_path.parent.stem

    full_output_path.mkdir(parents=True, exist_ok=True)

    perturbed_audio_path = full_output_path / (
        audio_path.stem + f"_sp{perturbation}" + audio_path.suffix
    )

    if perturbed_audio_path.exists():
        print(f"Perturbed audio already exists at {perturbed_audio_path}, skipping processing.")
        new_sample = sample.copy()
        new_sample["audio_filepath"] = str(perturbed_audio_path)
        new_sample["perturbation"] = perturbation
        return new_sample

    audio = librosa.load(str(audio_path), sr=sample["samplerate"])[0]

    resampled_audio = librosa.resample(
        audio, orig_sr=sample["samplerate"], target_sr=int(sample["samplerate"] * perturbation)
    )

    sf.write(perturbed_audio_path, resampled_audio, int(sample["samplerate"] * perturbation))

    new_sample = sample.copy()
    new_sample["audio_filepath"] = str(perturbed_audio_path.resolve())

    new_sample["perturbation"] = perturbation

    return new_sample


def process_manifest(
    manifest_path: Path,
    manifest_output_path: Path,
    audio_output_path: Path,
    perturbations: list[float],
):
    """Process the manifest file and apply speed perturbations.

    Args:
        manifest_path (str): Path to the input manifest file.
        output_path (str): Path to the output manifest file.
        perturbations (list): List of speed perturbation factors.
    """
    manifest = load_manifest(manifest_path)

    audio_output_path.mkdir(parents=True, exist_ok=True)

    new_manifest = []

    for sample in tqdm(manifest):
        for perturbation in perturbations:
            new_sample = process_sample(sample, audio_output_path, perturbation)
            new_manifest.append(new_sample)

    output_manifest_path = manifest_output_path / manifest_path.name

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for entry in new_manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"Processed manifest saved to {output_manifest_path}")


def process_cv_manifest_dir(manifest_dir: Path, output_path: Path, perturbations: list[float]):
    """Process the manifest file and apply speed perturbations.

    Args:
        manifest_dir (Path): Path to the input manifest directory.
        output_path (Path): Path to the output manifest file.
        perturbations (list): List of speed perturbation factors.
    """
    train_manifest_files = list(manifest_dir.glob("**/train_manifest.jsonl"))

    for train_manifest_file in train_manifest_files:
        relative_path = train_manifest_file.relative_to(manifest_dir.parent)
        output_manifest_dir = output_path / relative_path.parent

        output_manifest_dir.mkdir(parents=True, exist_ok=True)

        process_manifest(train_manifest_file, output_manifest_dir, output_path, perturbations)

    # Copy test manifests
    test_manifest_files = list(manifest_dir.glob("**/test_manifest.jsonl"))
    for test_manifest_file in test_manifest_files:
        relative_path = test_manifest_file.relative_to(manifest_dir.parent)
        output_manifest_dir = output_path / relative_path.parent

        output_manifest_dir.mkdir(parents=True, exist_ok=True)
        output_manifest_path = output_manifest_dir / test_manifest_file.name
        with (
            open(test_manifest_file, "r", encoding="utf-8") as src_f,
            open(output_manifest_path, "w", encoding="utf-8") as dst_f,
        ):
            for line in src_f:
                dst_f.write(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speed perturbation for NeMo dataset manifest.")
    parser.add_argument(
        "--manifest-path",
        required=False,
        help="Path to the input manifest file.",
        default=None,
    )
    parser.add_argument(
        "--manifest-dir",
        required=False,
        help="Path to the input manifest directory.",
        default=None,
    )
    parser.add_argument(
        "--output-path", type=Path, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--perturbations",
        type=float,
        nargs="+",
        default=[0.9, 1.0, 1.1],
        help="List of speed perturbation factors.",
    )

    args = parser.parse_args()

    if args.manifest_dir is not None:
        process_cv_manifest_dir(Path(args.manifest_dir), args.output_path, args.perturbations)
    elif args.manifest_path is not None:
        process_manifest(
            Path(args.manifest_path), args.output_path, args.output_path, args.perturbations
        )
