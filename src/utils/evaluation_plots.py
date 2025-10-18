import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(df, model=None, dataset_name=None, dataset_subset=None, dataset_split=None):

    sns.set(style="whitegrid")

    # filter DF to only rows where CER and WER are below 2.0
    df = df[(df["CER"] <= 2.0) & (df["WER"] <= 2.0)]

    # plot distribution of CER and WER and compute correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["CER"], bins=30, kde=True, ax=axes[0])
    if model and dataset_name and dataset_subset and dataset_split:
        axes[0].set_title(f"Distribution of CER for {model} on {dataset_name} ({dataset_subset}, {dataset_split})")
    else:
        axes[0].set_title("Distribution of CER")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Density")

    sns.histplot(df["WER"], bins=30, kde=True, ax=axes[1])
    if model and dataset_name and dataset_subset and dataset_split:
        axes[1].set_title(f"Distribution of WER for {model} on {dataset_name} ({dataset_subset}, {dataset_split})")
    else:
        axes[1].set_title("Distribution of WER")
    axes[1].set_xlabel("WER")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    if model and dataset_name and dataset_subset and dataset_split:
        Path(f"reports/figures/{model}/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"reports/figures/{model}/{dataset_name}_{dataset_subset}_{dataset_split}_wer_distribution.png")
    else:
        plt.savefig(f"reports/figures/wer_distribution.png")

    # plot CER and WER vs clip length
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x="clip_length", y="CER", data=df, ax=axes[0])
    if model and dataset_name and dataset_subset and dataset_split:
        axes[0].set_title(f"CER vs Clip Length for {model} on {dataset_name} ({dataset_subset}, {dataset_split})")
    else:
        axes[0].set_title("CER vs Clip Length")
    axes[0].set_xlabel("Clip Length (s)")
    axes[0].set_ylabel("CER")
    sns.scatterplot(x="clip_length", y="WER", data=df, ax=axes[1])
    if model and dataset_name and dataset_subset and dataset_split:
        axes[1].set_title(f"WER vs Clip Length for {model} on {dataset_name} ({dataset_subset}, {dataset_split})")
    else:
        axes[1].set_title("WER vs Clip Length")
    axes[1].set_xlabel("Clip Length (s)")
    axes[1].set_ylabel("WER")
    plt.tight_layout()
    if model and dataset_name and dataset_subset and dataset_split:
        Path(f"reports/figures/{model}/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"reports/figures/{model}/{dataset_name}_{dataset_subset}_{dataset_split}_clip_length.png")
    else:
        plt.savefig(f"reports/figures/clip_length.png")


def plot_semantic_distance(df, model=None, dataset=None, subset=None, split=None, save=True):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="semantic_distance", y="WER", data=df)
    if model and dataset and subset and split:
        plt.title(f"Semantic Distance vs WER for {model} on {dataset} ({subset}, {split})")
    else:
        plt.title("Semantic Distance vs WER")
    plt.xlabel("SemDist (lower is better)")
    plt.ylabel("WER")
    plt.tight_layout()

    Path(f"reports/figures/{model}/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"reports/figures/{model}/{dataset}_{subset}_{split}_semantic_distance.png", dpi=200, bbox_inches="tight")
    plt.close()


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

    
    Path(f"reports/figures/{model}/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"reports/figures/{model}/{dataset}_{subset}_{split}_correlation_matrix.png", dpi=200, bbox_inches="tight")
    
    plt.close()