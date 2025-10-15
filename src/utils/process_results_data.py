"""Evaluation of ASR models."""

import uuid

from dotenv import load_dotenv
from evaluate.loading import load as load_metric
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import pandas as pd
from pathlib import Path
import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

from utils.config_schema import ConfigSchema

load_dotenv()


def process_results_data(config: ConfigSchema) -> dict[str, float]:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    logger.info(f"Loading the results for {config.model.name} on {config.dataset.name}...")
    results_df = load_latest_detailed_results_parsed(
        model=config.model.name,
        dataset=config.dataset.name,
        subset=config.dataset.dataset_subset,
        split=config.dataset.eval_split_name,
    )

    logger.info("Computing the scores...")
    metrics = compute_avg_metrics(results_df, 
        model=config.model.name,
        dataset=config.dataset.name,
        subset=config.dataset.dataset_subset,
        split=config.dataset.eval_split_name,
    )

    logger.info("Computed average metrics.")

    logger.info("Saving metrics to JSON...")
    metrics_path = Path(
    f"reports/metrics/{config.model.name}_{config.dataset.name}_{config.dataset.dataset_subset}_{config.dataset.eval_split_name}_metrics.json"
)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to reports/metrics/")
    logger.info("Plotting the metrics...")
    # Plot the metrics
    plot_metrics(
        results_df,
        model=config.model.name,
        dataset=config.dataset.name,
        subset=config.dataset.dataset_subset,
        split=config.dataset.eval_split_name,
    )
    logger.info("Plots saved to reports/figures/")

    return metrics


# metrics may already be dicts or strings that look like dicts
def to_dict_safe(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    return ast.literal_eval(x if isinstance(x, str) else str(x))

def get_path_to_latest_detailed_results_parquet(model, dataset, subset, split, base="experiments/evaluate_model"):
    root = Path(base) / f"{model}_{dataset}_{subset}_{split}"

    # find every detailed_results.parquet under root, ignore .hydra
    files = [p for p in root.rglob("detailed_results.parquet") if ".hydra" not in p.parts]
    if not files:
        raise FileNotFoundError(f"No detailed_results.parquet under {root}")

    newest = max(files, key=lambda p: p.stat().st_mtime)
    return newest

def load_latest_detailed_results_parsed(model, dataset, subset, split, base="experiments/evaluate_model"):
    """
    Returns (df, parquet_path)

    df columns: id, prediction, label, clip_length, CER, WER
    """
    newest = get_path_to_latest_detailed_results_parquet(model, dataset, subset, split, base)

    # read with pyarrow to avoid partial row group reads
    df = pd.read_parquet(newest, engine="pyarrow")

    m = df["metrics"].apply(to_dict_safe)
    metrics_df = pd.json_normalize(m)

    # attach CER and WER, drop original metrics
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1).rename(
        columns={"cer": "CER", "wer": "WER"}
    )

    return df


def load_results_json_for_config(model, dataset, subset, split, base="experiments/evaluate_model"):
    """
    Return dict loaded from the newest results.json for this config.
    """
    root = Path(base) / f"{model}_{dataset}_{subset}_{split}"
    files = [p for p in root.rglob("results.json") if ".hydra" not in p.parts]
    if not files:
        return None, None
    path = max(files, key=lambda p: p.stat().st_mtime)
    with open(path, "r") as f:
        data = json.load(f)
    return data, path


def compute_avg_metrics(df, model=None, dataset=None, subset=None, split=None):
    metrics = {}
    
    # compute WER and CER again (load from evaluate to be sure)
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    metrics["WER"] = wer_metric.compute(predictions=df["prediction"].tolist(), references=df["label"].tolist())
    metrics["CER"] = cer_metric.compute(predictions=df["prediction"].tolist(), references=df["label"].tolist())

    # compute median CER and WER
    metrics["CER_median"] = df["CER"].median()
    metrics["WER_median"] = df["WER"].median()

    # compute stddev CER and WER
    metrics["CER_std"] = df["CER"].std()
    metrics["WER_std"] = df["WER"].std()

    # compuute standard error of the mean (SEM) for CER and WER
    metrics["CER_sem"] = df["CER"].sem()
    metrics["WER_sem"] = df["WER"].sem()

    # compute average clip length
    metrics["avg_clip_length"] = df["clip_length"].mean()

    # do the same for all clips shorther than 10 seconds and longer than 0.5 seconds
    short_df = df[(df["clip_length"] < 10.0) & (df["clip_length"] > 0.5)]
    logger.info(f"Number of short clips (0.5s < clip_length < 10s): {len(short_df)}")
    metrics["CER_short"] = cer_metric.compute(predictions=short_df["prediction"].tolist(), references=short_df["label"].tolist())
    metrics["WER_short"] = wer_metric.compute(predictions=short_df["prediction"].tolist(), references=short_df["label"].tolist())
    metrics["CER_short_median"] = short_df["CER"].median()
    metrics["WER_short_median"] = short_df["WER"].median()
    metrics["CER_short_std"] = short_df["CER"].std()
    metrics["WER_short_std"] = short_df["WER"].std()
    metrics["CER_short_sem"] = short_df["CER"].sem()
    metrics["WER_short_sem"] = short_df["WER"].sem()
    metrics["avg_clip_length_short"] = short_df["clip_length"].mean()

    # add values from results.json for this config, if present
    if model and dataset and subset and split:
        results_json, _ = load_results_json_for_config(model, dataset, subset, split)
        if results_json:
            # keep your computed values as primary, but add the json for traceability
            if "rtf" in results_json:
                metrics["RTF"] = float(results_json["rtf"])
            if "rtfx" in results_json:
                metrics["RTFx"] = float(results_json["rtfx"])

    # format all floats to 4 decimal places
    if model and dataset and subset and split:
        metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        metrics = {'model': model, 'dataset': dataset, 'subset': subset, 'split': split, 'metrics': metrics}
    else:
        metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    return metrics

def plot_metrics(df, model=None, dataset=None, subset=None, split=None):

    sns.set(style="whitegrid")

    # plot distribution of CER and WER and compute correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["CER"], bins=30, kde=True, ax=axes[0])
    if model and dataset and subset and split:
        axes[0].set_title(f"Distribution of CER for {model} on {dataset} ({subset}, {split})")
    else:
        axes[0].set_title("Distribution of CER")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Density")

    sns.histplot(df["WER"], bins=30, kde=True, ax=axes[1])
    if model and dataset and subset and split:
        axes[1].set_title(f"Distribution of WER for {model} on {dataset} ({subset}, {split})")
    else:
        axes[1].set_title("Distribution of WER")
    axes[1].set_xlabel("WER")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    if model and dataset and subset and split:
        plt.savefig(f"reports/figures/{model}_{dataset}_{subset}_{split}_Distribution.png")
    else:
        plt.savefig(f"reports/figures/Distribution.png")

    # plot CER and WER vs clip length
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x="clip_length", y="CER", data=df, ax=axes[0])
    if model and dataset and subset and split:
        axes[0].set_title(f"CER vs Clip Length for {model} on {dataset} ({subset}, {split})")
    else:
        axes[0].set_title("CER vs Clip Length")
    axes[0].set_xlabel("Clip Length (s)")
    axes[0].set_ylabel("CER")
    sns.scatterplot(x="clip_length", y="WER", data=df, ax=axes[1])
    if model and dataset and subset and split:
        axes[1].set_title(f"WER vs Clip Length for {model} on {dataset} ({subset}, {split})")
    else:
        axes[1].set_title("WER vs Clip Length")
    axes[1].set_xlabel("Clip Length (s)")
    axes[1].set_ylabel("WER")
    plt.tight_layout()
    if model and dataset and subset and split:
        plt.savefig(f"reports/figures/{model}_{dataset}_{subset}_{split}_Clip_Length.png")
    else:
        plt.savefig(f"reports/figures/Clip_Length.png")
    

    # compute correlation between CER, WER and clip length
    corr = df[["CER", "WER", "clip_length"]].corr()
    plot_corr_mat(corr, model, dataset, subset, split)
    

def plot_corr_mat(corr, model=None, dataset=None, subset=None, split=None, save=True):

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, square=True, cbar=True
    )
    if model and dataset and subset and split:
        plt.title(f"Correlation Matrix for {model} on {dataset} ({subset}, {split})")
    else:
        plt.title("Correlation Matrix")
    plt.tight_layout()

    if save:
        plt.savefig(f"reports/figures/{model}_{dataset}_{subset}_{split}_correlation_matrix.png", dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

