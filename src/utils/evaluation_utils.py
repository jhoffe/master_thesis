import ast
import json
from pathlib import Path
import re
from typing import Dict, List

from carbontracker import parser
from evaluate.loading import load as load_metric
from loguru import logger
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELS = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "roest-wav2vec2-315m-v2",
    "roest-wav2vec2-1B-v2",
    "roest-wav2vec2-2B-v2",
    "hviske-v3-conversation",
    "hviske-v2",
    "roest-whisper-large-v1",
    "seamless-m4t-v2-large",
    "parakeet-tdt-0.6b-v3",
    "canary-1b-v2",
]

DATASETS = [
    "coral-v2",
    "fleurs",
]

SUBSETS = {
    "coral-v2": "read_aloud",
    "fleurs": "da_dk",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
}

SENTENCE_TRANSFORMER_MODEL = "KennethTM/MiniLM-L6-danish-encoder"


def provide_eval_combinations(
    models: List[str], datasets: List[str], subsets: dict, splits: dict
) -> List[dict]:
    return [
        {
            "model": model,
            "dataset_name": dataset,
            "dataset_subset": subsets[dataset],
            "dataset_split": splits[dataset],
        }
        for model in models
        for dataset in datasets
    ]


def filter_eval_grid(
    df: pd.DataFrame,
    models: List[str] = MODELS,
    datasets: List[str] = DATASETS,
    subsets: Dict[str, str] = SUBSETS,
    splits: Dict[str, str] = SPLITS,
) -> pd.DataFrame:
    """
    Return df filtered to the exact evaluation combinations you specified.
    Enforces categorical ordering for model and dataset_name.
    """
    eval_combos = [
        {
            "model": m,
            "dataset_name": d,
            "dataset_subset": subsets[d],
            "dataset_split": splits[d],
        }
        for m in models
        for d in datasets
    ]
    key_cols = ["model", "dataset_name", "dataset_subset", "dataset_split"]
    merged = df.merge(pd.DataFrame(eval_combos), on=key_cols, how="inner").copy()
    merged["model"] = pd.Categorical(merged["model"], categories=models, ordered=True)
    merged["dataset_name"] = pd.Categorical(
        merged["dataset_name"], categories=datasets, ordered=True
    )
    return merged


def to_dict_safe(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    return ast.literal_eval(x if isinstance(x, str) else str(x))


def get_path_to_latest_detailed_results_parquet(
    eval_combination: dict, base="experiments/evaluate_model"
) -> Path:
    """
    Returns the path to the latest detailed_results.parquet for the given eval combination.
    eval_combination: str
        A string of the form "{model}_{dataset}_{subset}_{split}"
    base: str
        The base directory where evaluation results are stored.
    """
    root = (
        Path(base)
        / f"{eval_combination['model']}_{eval_combination['dataset_name']}_{eval_combination['dataset_subset']}_{eval_combination['dataset_split']}"
    )

    # find every detailed_results.parquet under root, ignore .hydra
    files = [p for p in root.rglob("detailed_results.parquet") if ".hydra" not in p.parts]
    if not files:
        raise FileNotFoundError(f"No detailed_results.parquet under {root}")

    newest = max(files, key=lambda p: p.stat().st_mtime)
    return newest


def load_latest_detailed_results_parsed(eval_combination: dict, base="experiments/evaluate_model"):
    """
    Returns (df, parquet_path)

    df columns: id, prediction, label, clip_length, CER, WER
    """
    newest = get_path_to_latest_detailed_results_parquet(eval_combination, base)

    # read with pyarrow to avoid partial row group reads
    df = pd.read_parquet(newest, engine="pyarrow")

    m = df["metrics"].apply(to_dict_safe)
    metrics_df = pd.json_normalize(m)

    # attach CER and WER, drop original metrics
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1).rename(
        columns={"cer": "CER", "wer": "WER"}
    )

    # add model, dataset_name, dataset_subset, dataset_split columns
    df["model"] = eval_combination["model"]
    df["dataset_name"] = eval_combination["dataset_name"]
    df["dataset_subset"] = eval_combination["dataset_subset"]
    df["dataset_split"] = eval_combination["dataset_split"]

    # For FLEUERS, the 'id' column is not actuallys id's but rather utterance indices.
    # Therefore, we remove the 'id' column to avoid confusion.
    if eval_combination["dataset_name"] == "fleurs":
        df = df.drop(columns=["id"])

        # next, we replace it with a new 'id' column that is simply a range from 1 to len(df)
        ids = [f"rec_{idx}" for idx in range(1, len(df) + 1)]
        df["id"] = ids

    if eval_combination["dataset_name"] == "coral-v2":
        pitch_data = pd.read_parquet("reports/metrics/coral-v2-summary.parquet")
        # drop dataset_name column and clip_length column
        pitch_data = pitch_data.drop(columns=["dataset_name", "clip_length"])
        # rename id_recording to id in pitch_data
        pitch_data = pitch_data.rename(columns={"id_recording": "id"})
        df = df.merge(pitch_data, on="id", how="left")
    elif eval_combination["dataset_name"] == "fleurs":
        pitch_data = pd.read_parquet("reports/metrics/fleurs-summary.parquet")
        # drop unneeded columns
        colums_to_drop = [
            "id",
            "gender",
            "dataset_name",
            "clip_length",
            "path",
            "num_samples",
            "transcription",
            "raw_transcription",
            "language",
            "lang_group_id",
            "lang_id",
        ]
        pitch_data = pitch_data.drop(columns=colums_to_drop)
        # rename id_recording to id in pitch_data
        pitch_data = pitch_data.rename(columns={"id_recording": "id"})

        df = df.merge(pitch_data, on="id", how="left")

    # compute number of words in label column
    # df["num_words"] = df["label"].apply(lambda x: len(x.split()))

    # compute number of words per second
    # df["words_per_sec"] = df["num_words"] / df["clip_length"]

    return df


def load_results_json_for_config(
    eval_combination: dict, base="experiments/evaluate_model"
) -> dict | None:
    """
    Return dict loaded from the newest results.json for this config.
    """
    root = (
        Path(base)
        / f"{eval_combination['model']}_{eval_combination['dataset_name']}_{eval_combination['dataset_subset']}_{eval_combination['dataset_split']}"
    )
    files = [p for p in root.rglob("results.json") if ".hydra" not in p.parts]
    if not files:
        return None
    path = max(files, key=lambda p: p.stat().st_mtime)
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_carbon_log(eval_combination: dict, base="carbon_logs") -> dict:
    """
    Load the latest carbon log for the given model/dataset/subset/split.

    Returns:
        A dictionary with the carbon log data.
    """
    model = eval_combination["model"]
    dataset_name = eval_combination["dataset_name"]
    dataset_subset = eval_combination["dataset_subset"]
    dataset_split = eval_combination["dataset_split"]

    logs = parser.parse_all_logs(log_dir=base)

    pattern = re.compile(
        r"eval-(?P<model>[^-]+)-(?P<dataset_name>[^-]+)-(?P<dataset_subset>[^-]+)-(?P<dataset_split>[^_]+)_"
    )

    target_prefix = f"eval-{model}-{dataset_name}-{dataset_subset}-{dataset_split}"

    matching_logs = [
        log
        for log in logs
        if re.search(pattern, log["output_filename"]) and target_prefix in log["output_filename"]
    ]

    if matching_logs:
        selected_log = matching_logs[-1]  # get the latest log
        return selected_log
    else:
        logger.warning(f"No carbon log found for {target_prefix}")
        return None


def combine_all_detailed_results(
    eval_combination: List[dict], base="experiments/evaluate_model"
) -> pd.DataFrame:
    """Load and combine all detailed results for the given eval combination.

    Args:
        eval_combination:
            A dictionary with keys 'model', 'dataset_name', 'dataset_subset', 'dataset_split'.
        base:
            The base directory where evaluation results are stored.

    Returns:
        Combined DataFrame.
    """
    dfs = []
    for eval_combination in eval_combination:
        df = load_latest_detailed_results_parsed(eval_combination=eval_combination, base=base)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def save_to_parquet(df: pd.DataFrame, base_path: Path, file_name: str) -> Path:
    """Save the detailed results DataFrame to a parquet file.

    Args:
        df:
            DataFrame with detailed results.
        save_path:
            Path to save the parquet file.

    Returns:
        Path to the saved parquet file.
    """
    base_path.mkdir(parents=True, exist_ok=True)
    save_path = base_path / file_name
    df.to_parquet(save_path)
    return save_path


def load_from_parquet(parquet_path: Path) -> pd.DataFrame:
    """Load a DataFrame from a parquet file.

    Args:
        parquet_path:
            Path to the parquet file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_parquet(parquet_path)


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load a DataFrame from a CSV file.

    Args:
        csv_path:
            Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(csv_path)


def _semantic_distance(label, prediction):
    return 1 - np.dot(label, prediction) / (np.linalg.norm(label) * np.linalg.norm(prediction))


def compute_sentence_embeddings(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute sentence embeddings for the predictions and labels in the DataFrame.

    Args:
        df:
            DataFrame with columns 'prediction' and 'label'.
        model_name:
            The name of the model to use for computing embeddings.

    Returns:
        DataFrame with additional columns 'prediction_embedding' and 'label_embedding'.
    """

    logger.info(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    predictions = df["prediction"].tolist()
    labels = df["label"].tolist()

    logger.info("Computing embeddings for predictions...")
    pred_embeddings = model.encode(predictions, normalize_embeddings=True, batch_size=256)

    logger.info("Computing embeddings for labels...")
    label_embeddings = model.encode(labels, normalize_embeddings=True, batch_size=256)

    df["prediction_embedding"] = list(pred_embeddings)
    df["label_embedding"] = list(label_embeddings)

    logger.info("Embeddings computed.")

    # finally, we need the semantic distance between prediction_embedding and label_embedding
    df["semantic_distance"] = df.apply(
        lambda x: _semantic_distance(x["label_embedding"], x["prediction_embedding"]), axis=1
    )
    return df


def compute_avg_metrics(df: pd.DataFrame, eval_combination: dict) -> pd.DataFrame:
    metrics = {}

    # compute WER and CER again (load from evaluate to be sure)
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    metrics["WER"] = wer_metric.compute(
        predictions=df["prediction"].tolist(), references=df["label"].tolist()
    )
    metrics["CER"] = cer_metric.compute(
        predictions=df["prediction"].tolist(), references=df["label"].tolist()
    )

    # compute median CER and WER
    metrics["CER_median"] = df["CER"].median()
    metrics["WER_median"] = df["WER"].median()

    # compute stddev CER and WER
    metrics["CER_std"] = df["CER"].std()
    metrics["WER_std"] = df["WER"].std()

    # compuute standard error of the mean (SEM) for CER and WER
    metrics["CER_sem"] = df["CER"].sem()
    metrics["WER_sem"] = df["WER"].sem()

    # compute average semantic distance (cosine similarity)
    metrics["avg_semantic_distance"] = df["semantic_distance"].mean()
    metrics["semantic_distance_median"] = df["semantic_distance"].median()
    metrics["semantic_distance_std"] = df["semantic_distance"].std()
    metrics["semantic_distance_sem"] = df["semantic_distance"].sem()

    # compute average clip length
    metrics["avg_clip_length"] = df["clip_length"].mean()

    # add values from results.json for this config, if present
    results_json = load_results_json_for_config(eval_combination)
    if results_json:
        if "rtf" in results_json:
            metrics["RTF"] = float(results_json["rtf"])
        if "rtfx" in results_json:
            metrics["RTFx"] = float(results_json["rtfx"])

    # add values from carbon log, if present
    carbon_log = load_carbon_log(eval_combination)
    if carbon_log:
        metrics["co2_g"] = carbon_log["actual"]["co2eq (g)"]
        metrics["energy_kWh"] = carbon_log["actual"]["energy (kWh)"]
        metrics["duration_s"] = carbon_log["actual"]["duration (s)"]

    metrics_df = pd.DataFrame([metrics])
    return metrics_df


def compute_average_metrics_for_detailed_results(
    df: pd.DataFrame, eval_combinations: List[dict]
) -> dict:
    """Compute average metrics for the detailed results DataFrame.

    Args:
        df:
            DataFrame with detailed results.

    Returns:
        Dictionary with average metrics.
    """
    avg_metrics_list = []
    for eval_combination in eval_combinations:
        logger.info(f"Computing average metrics for {eval_combination}...")
        subset_df = df[
            (df["model"] == eval_combination["model"])
            & (df["dataset_name"] == eval_combination["dataset_name"])
            & (df["dataset_subset"] == eval_combination["dataset_subset"])
            & (df["dataset_split"] == eval_combination["dataset_split"])
        ]
        if not subset_df.empty:
            avg_metrics_df = compute_avg_metrics(subset_df, eval_combination)
            avg_metrics_df["model"] = eval_combination["model"]
            avg_metrics_df["dataset_name"] = eval_combination["dataset_name"]
            avg_metrics_df["dataset_subset"] = eval_combination["dataset_subset"]
            avg_metrics_df["dataset_split"] = eval_combination["dataset_split"]
            avg_metrics_list.append(avg_metrics_df)
    combined_avg_metrics_df = pd.concat(avg_metrics_list, ignore_index=True)
    return combined_avg_metrics_df
