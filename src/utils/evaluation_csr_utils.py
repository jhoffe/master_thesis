import ast
import json
import logging
from pathlib import Path
import re
from typing import Dict, List
import uuid

from carbontracker import parser
import datasets
import evaluate
from evaluate.loading import load as load_metric
from loguru import logger
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import transformers

MODELS = [
    "parakeet_finetune_spec-aug",
    "parakeet-finetune_SA_ll",
    "parakeet-finetune_SA_ll_SA",
    "parakeet-finetune_SA_ll_PS",
    "parakeet-finetune_SA_ll_SP",
    "parakeet-finetune_SA_ll_SA_PS",
    "parakeet-finetune_SA_ll_SA_SP",
    "parakeet-finetune_SA_ll_PS_SP",
    "parakeet-finetune_SA_ll_SA_PS_SP",
    "canary_finetune_spec-aug_speed-perturbations",
]

MODELS = [
    "parakeet-finetune_SA_ll",
    "parakeet-finetune_SA_ll_SA",
    "parakeet-finetune_SA_ll_PS",
    "parakeet-finetune_SA_ll_SP",
    "parakeet-finetune_SA_ll_SA_PS",
    "parakeet-finetune_SA_ll_SA_SP",
    "parakeet-finetune_SA_ll_PS_SP",
    "parakeet-finetune_SA_ll_SA_PS_SP",
]

DATASETS = [
    "coral-v2",
    "fleurs",
    "lillelyd"
]

SUBSETS = {
    "coral-v2": "read_aloud",
    "fleurs": "da_dk",
    "lillelyd": "full",
}

CV_FOLDS = {
    "cv-1",
    "cv-2",
    "cv-3",
    "cv-4",
}

SPLITS = {
    "coral-v2": "test",
    "fleurs": "test",
    "lillelyd": "test",
}

SENTENCE_TRANSFORMER_MODEL = "KennethTM/MiniLM-L6-danish-encoder"

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
evaluate.logging.set_verbosity_error()
logging.getLogger("evaluate").setLevel(logging.ERROR)


def to_dict_safe(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    return ast.literal_eval(x if isinstance(x, str) else str(x))


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


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
    if eval_combination["cv_fold"] is None:
        root = (
            Path(base)
            / f"{eval_combination['model']}_{eval_combination['dataset_name']}_{eval_combination['dataset_subset']}_{eval_combination['dataset_split']}"
        )
    else:
        root = (
            Path(base)
            / f"{eval_combination['model']}_{eval_combination['cv_fold']}_{eval_combination['dataset_name']}_{eval_combination['dataset_subset']}_{eval_combination['dataset_split']}"
        )

    # find every detailed_results.parquet under root, ignore .hydra
    files = [p for p in root.rglob("detailed_results.parquet") if ".hydra" not in p.parts]
    if not files:
        raise FileNotFoundError(f"No detailed_results.parquet under {root}")

    newest = max(files, key=lambda p: p.stat().st_mtime)
    return newest

def _semantic_distance(label, prediction):
    return 1 - np.dot(label, prediction) / (np.linalg.norm(label) * np.linalg.norm(prediction))

def compute_sentence_embeddings(df: pd.DataFrame, st_model) -> pd.DataFrame:
    """Compute sentence embeddings for the predictions and labels in the DataFrame.

    Args:
        df:
            DataFrame with columns 'prediction' and 'label'.
        model_name:
            The name of the model to use for computing embeddings.

    Returns:
        DataFrame with additional columns 'prediction_embedding' and 'label_embedding'.
    """

    #logger.info(f"Loading sentence transformer model: {model_name}")
    model = st_model

    predictions = df["prediction"].tolist()
    labels = df["label"].tolist()

    #logger.info("Computing embeddings for predictions...")
    pred_embeddings = model.encode(predictions, normalize_embeddings=True, batch_size=256)

    #logger.info("Computing embeddings for labels...")
    label_embeddings = model.encode(labels, normalize_embeddings=True, batch_size=256)

    df["prediction_embedding"] = list(pred_embeddings)
    df["label_embedding"] = list(label_embeddings)

    #logger.info("Embeddings computed.")

    # finally, we need the semantic distance between prediction_embedding and label_embedding
    df["semantic_distance"] = df.apply(
        lambda x: _semantic_distance(x["label_embedding"], x["prediction_embedding"]), axis=1
    )
    return df


def load_latest_detailed_results_parsed(eval_combination: dict, base="experiments/evaluate_model", st_model=None) -> pd.DataFrame:
    """
    Returns (df, parquet_path)

    df columns: id, prediction, label, clip_length, CER, WER
    """
    try:
        newest = get_path_to_latest_detailed_results_parquet(eval_combination, base)
    except FileNotFoundError:
        return None

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
    df["cv_fold"] = eval_combination["cv_fold"]

    # attach embeddings and semantic distance
    df = compute_sentence_embeddings(df, st_model=st_model)

    # check for lillelyd
    if eval_combination["dataset_name"] == "lillelyd":
        # for lillelyd, we split the id on "/" and take the two last parts and join them with "/"
        df["id"] = df["id"].apply(lambda x: "/".join(Path(x).parts[-2:]))
        pitch_data = pd.read_parquet("reports/metrics/lillelyd-summary.parquet")
        # drop unneeded columns
        columns_to_drop = [
            "id_recording",
            "dataset_name",
            "clip_length"
        ]
        pitch_data = pitch_data.drop(columns=columns_to_drop)
        # rename audio_filepath to id in pitch_data
        pitch_data = pitch_data.rename(columns={"audio_filepath": "id"})
        df = df.merge(pitch_data, on="id", how="left")

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

    return df


def combine_all_detailed_results_lillelyd(combinations: list[dict], base="experiments/evaluate_model") -> pd.DataFrame:
    temp_dfs = []
    final_dfs = []

    # 2. Initialize the embedding model ONCE outside the loop
    print("Initializing SentenceTransformer...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 3. Use a single tqdm bar with a description
    pbar = tqdm(combinations, desc="Processing evaluations")
    for combo in pbar:
        pbar.set_postfix({"model": combo["model"], "dataset": combo['dataset_name'], "fold": combo['cv_fold']})
        
        # Pass the pre-loaded model to the loader to avoid re-init overhead
        df = load_latest_detailed_results_parsed(combo, base=base, st_model=st_model)
        
        if df is None or df.empty:
            print(f"Warning: No data for combination {combo}")
            continue

        if combo["dataset_name"] == "lillelyd":
            final_dfs.append(df)
        else:
            temp_dfs.append(df)

    if temp_dfs:
        combined_temp_df = pd.concat(temp_dfs, ignore_index=True)
        numeric_cols = combined_temp_df.select_dtypes(include=['number']).columns.tolist()

        group_cols = ["id", "model", "dataset_name", "dataset_subset", "dataset_split"]
        non_numeric_cols = [col for col in combined_temp_df.columns if col not in numeric_cols and col not in group_cols]
        agg_map = {col: 'mean' for col in numeric_cols if col != 'cv_fold'}
        for col in non_numeric_cols:
            if col in combined_temp_df.columns: agg_map[col] = "first"
            
        averaged_df = combined_temp_df.groupby(group_cols).agg(agg_map).reset_index()
        averaged_df["cv_fold"] = "averaged"
        final_dfs.append(averaged_df)

    full_df = pd.concat(final_dfs, ignore_index=True)
    full_df['age'] = full_df['age'].astype(str)
        
    return full_df

def make_stitched_lillelyd_df(
    combined_df: pd.DataFrame,
    *,
    dedup_lillelyd_on_id: bool = True,
    drop_cv_fold_column: bool = True,
) -> pd.DataFrame:
    """
    Returns a version where LilleLyd cv folds are stitched (concatenated) into one dataset.

    LilleLyd:
      - concatenates rows from cv-1..cv-4
      - optionally de-duplicates by id per (model, dataset metadata)
      - drops cv_fold (or you can keep it by setting drop_cv_fold_column=False)

    coral-v2 and fleurs:
      - unchanged (typically cv_fold == 'averaged' in your pipeline)
    """

    required = {"dataset_name", "model", "dataset_subset", "dataset_split"}
    missing = required - set(combined_df.columns)
    if missing:
        raise ValueError(f"combined_df missing required columns: {sorted(missing)}")

    df = combined_df.copy()

    # Split
    is_lillelyd = df["dataset_name"].astype(str).str.lower().eq("lillelyd")
    lille = df[is_lillelyd].copy()
    other = df[~is_lillelyd].copy()

    # Stitch LilleLyd: just concatenate folds (already in one df), then optionally dedup
    if dedup_lillelyd_on_id:
        if "id" not in lille.columns:
            raise ValueError("Cannot deduplicate LilleLyd without an 'id' column.")
        key_cols = ["id", "model", "dataset_name", "dataset_subset", "dataset_split"]
        lille = (
            lille.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="first")
            .reset_index(drop=True)
        )

    # Drop fold for stitched version
    if drop_cv_fold_column and "cv_fold" in lille.columns:
        lille = lille.drop(columns=["cv_fold"])

    # If you drop cv_fold for lillelyd but keep it for others, schemas differ.
    # Usually you either drop it for all or keep it with a constant value for lillelyd.
    if drop_cv_fold_column and "cv_fold" in other.columns:
        other = other.drop(columns=["cv_fold"])

    stitched = pd.concat([lille, other], ignore_index=True)
    return stitched

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

    # compute {metric}_ci_lower and {metric}_ci_upper for CER and WER
    metrics["CER_ci_lower"] = metrics["CER"] - 1.96 * metrics["CER_sem"]
    metrics["CER_ci_upper"] = metrics["CER"] + 1.96 * metrics["CER_sem"]
    metrics["WER_ci_lower"] = metrics["WER"] - 1.96 * metrics["WER_sem"]
    metrics["WER_ci_upper"] = metrics["WER"] + 1.96 * metrics["WER_sem"]

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
    for eval_combination in tqdm(eval_combinations):
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
