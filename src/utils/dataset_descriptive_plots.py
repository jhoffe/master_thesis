from utils.evaluation_utils import (
    load_from_parquet,
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from loguru import logger


sns.set(style="whitegrid")


DATASETS = {
    "fleurs": "google--fleurs-da_dk-test-unfiltered",
    "coral-v2": "CoRal-project--coral-v2-read_aloud-test-unfiltered",
}


METRICS = [
    "clip_length",
    "mean_pitch_hz",
    "median_pitch_hz",
    "voiced_ratio",
]


FORMAT_DICT = {
    "dataset_name": "Dataset",
    "coral-v2": "CoRal-v2 Read Aloud Testset",
    "fleurs": "Fleurs da_dk Testset",
    "clip_length": "Clip Length (s)",
    "mean_pitch_hz": "Mean Pitch (Hz)",
    "median_pitch_hz": "Median Pitch (Hz)",
    "voiced_ratio": "Voiced Ratio",
}


def _fmt(label: str) -> str:
    return FORMAT_DICT.get(label, label)


def save_plot(base_path: str, filename: str) -> None:
    """
    Save the current matplotlib plot to the specified path.
    """
    Path(base_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{base_path}/{filename}", bbox_inches='tight', dpi=200)
    plt.close()

def distribution_plot(dataset_name: str, metric: str, base_path: str) -> None:
    """
    Generate and save distribution plots for dataset descriptive statistics.
    """
    df = load_from_parquet(f"reports/metrics/{dataset_name}-summary.parquet")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[metric], bins=30, kde=True)
    plt.title(f'{_fmt(metric)} distribution for {_fmt(dataset_name)}')
    plt.xlabel(_fmt(metric))    
    plt.ylabel('Frequency')
    save_plot(base_path=f'reports/plots/{dataset_name}', filename=f'{metric}_distribution.png')


def age_plot(dataset_name: str, base_path: str) -> None:
    """
    Generate and save age distribution plot for dataset descriptive statistics.
    """
    df = load_from_parquet(f"reports/metrics/{dataset_name}-summary.parquet")

    # make bins 0-9, 10-19, ..., 90-99
    df['age_bin'] = pd.cut(df['age'], bins=range(0, 101, 10), right=False)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='age_bin', data=df, stat='percent')
    plt.title(f'Age distribution for {_fmt(dataset_name)}', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    
    # make ticks show percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    total = len(df)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                           textcoords='offset points')
    save_plot(base_path=base_path, filename='age_distribution.png')


def age_plot_by_gender(dataset_name: str, base_path: str) -> None:
    """
    Generate and save age distribution grouped by gender (side-by-side bars per age bin).

    - Uses the same 10-year bins as age_plot
    - Displays percentages on the y-axis
    - Requires a 'gender' column; skips gracefully if missing
    """
    df = load_from_parquet(f"reports/metrics/{dataset_name}-summary.parquet")

    if 'gender' not in df.columns:
        logger.warning(f"'gender' column not found in {dataset_name}-summary.parquet. Skipping age_by_gender plot.")
        return

    # Drop missing genders and bin ages
    df = df[df['gender'].notna()].copy()
    df['age_bin'] = pd.cut(df['age'], bins=range(0, 101, 10), right=False)

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='age_bin', hue='gender', data=df, stat='percent')
    plt.title(f'Age distribution by gender for {_fmt(dataset_name)}', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)

    # Format y-axis as percent
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Annotate each bar with its percentage
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 4),
                textcoords='offset points'
            )

    plt.legend(title='Gender')
    save_plot(base_path=base_path, filename='age_distribution_by_gender.png')


def distribution_by_gender(dataset_name: str, base_path: str) -> None:
    """
    Generate and save distribution plots for each metric, split by gender.
    Expects a 'gender' column in the dataset summary parquet.

    Saves one PNG per metric to: reports/plots/{dataset_name}/{metric}_by_gender.png
    """
    df = load_from_parquet(f"reports/metrics/{dataset_name}-summary.parquet")

    if 'gender' not in df.columns:
        logger.warning(f"'gender' column not found in {dataset_name}-summary.parquet. Skipping gender plots.")
        return

    # Clean up gender values (optional but helps avoid NaNs showing up as a legend entry)
    df = df[df['gender'].notna()].copy()

    for metric in METRICS:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe for {dataset_name}. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df,
            x=metric,
            hue='gender',
            bins=30,
            kde=True,
            stat='percent',        # y axis in percent
            common_norm=False,     # each gender scaled independently
            element='step',        # cleaner overlay look
            alpha=0.4
        )

        plt.title(f"{_fmt(metric)} distribution by gender for {_fmt(dataset_name)}")
        plt.xlabel(_fmt(metric))
        plt.ylabel("Percentage")

        # Ensure output dir exists and save
        save_plot(base_path=f"reports/plots/{dataset_name}", filename=f"{metric}_by_gender.png")


def make_coral_plots() -> None:
    """
    Generate and save distribution plots for CoRal v2 dataset descriptive statistics.
    """
    dataset_name = "coral-v2"
    for metric in METRICS:
        logger.info(f"Generating distribution plot for {dataset_name} - {metric}...")
        distribution_plot(dataset_name, metric, base_path=f'reports/plots/{dataset_name}')

    # also make distribution plots for age
    logger.info(f"Generating age distribution plot for {dataset_name}...")
    age_plot(dataset_name, base_path=f'reports/plots/{dataset_name}')

    logger.info(f"Generating age distribution by gender plot for {dataset_name}...")
    age_plot_by_gender(dataset_name, base_path=f'reports/plots/{dataset_name}')

    logger.info(f"Generating gender distribution plots for {dataset_name}...")
    distribution_by_gender(dataset_name, base_path=f'reports/plots/{dataset_name}')


def make_fleurs_plots() -> None:
    """
    Generate and save distribution plots for Fleurs dataset descriptive statistics.
    """
    dataset_name = "fleurs"
    for metric in METRICS:
        logger.info(f"Generating distribution plot for {dataset_name} - {metric}...")
        distribution_plot(dataset_name, metric, base_path=f'reports/plots/{dataset_name}')