import os
from pathlib import Path
from typing import Dict, List, Optional

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from loguru import logger
from tqdm import tqdm

from utils.evaluation_utils import save_to_parquet, load_from_parquet

###################
## CONFIGURATION ##
###################
MODELS = {
    "Canary": [
        "canary_finetune",
        "canary_finetune_pitch-shift",
        "canary_finetune_spec-aug",
        "canary_finetune_speed-perturbations",
        "canary_finetune_spec-aug_pitch-shift",
        "canary_finetune_spec-aug_speed-perturbations",
        "canary_finetune_speed-perturbations_pitch-shift",
        "canary_finetune_spec-aug_speed-perturbations_pitch-shift",
    ],
    "Parakeet": [
        "parakeet_finetune",
        "parakeet_finetune_pitch-shift",
        "parakeet_finetune_spec-aug",
        "parakeet_finetune_speed-perturbations",
        "parakeet_finetune_spec-aug_pitch-shift",
        "parakeet_finetune_spec-aug_speed-perturbations",
        "parakeet_finetune_speed-perturbations_pitch-shift",
        "parakeet_finetune_spec-aug_speed-perturbations_pitch-shift",
    ]
}


FORMAT_DICT = {
    ('parakeet_finetune', 'canary_finetune'): 'FT',
    ('parakeet_finetune_spec-aug', 'canary_finetune_spec-aug'): 'FT+SA',
    ('parakeet_finetune_speed-perturbations', 'canary_finetune_speed-perturbations'): 'FT+SP',
    ('parakeet_finetune_pitch-shift', 'canary_finetune_pitch-shift'): 'FT+PS',
    ('parakeet_finetune_spec-aug_speed-perturbations', 'canary_finetune_spec-aug_speed-perturbations'): 'FT+SA+SP',
    ('parakeet_finetune_spec-aug_pitch-shift', 'canary_finetune_spec-aug_pitch-shift'): 'FT+SA+PS',
    ('parakeet_finetune_speed-perturbations_pitch-shift', 'canary_finetune_speed-perturbations_pitch-shift'): 'FT+SP+PS',
    ('parakeet_finetune_spec-aug_speed-perturbations_pitch-shift', 'canary_finetune_spec-aug_speed-perturbations_pitch-shift'): 'FT+SA+SP+PS'
}


###################
## HELPER FUNCS ###
###################

def _fmt(label: str, format_dict: Optional[Dict]=None) -> str:
    if format_dict is None:
        format_dict = FORMAT_DICT
    return format_dict.get(label, label)


def expand_dict(format_dict):
    expanded = {}
    for (m1, m2), label in format_dict.items():
        expanded[m1] = label
        expanded[m2] = label
    return expanded


####################
## CORE FUNCTIONS ##
####################

def build_scores_dict(df: pd.DataFrame, datasets: List[str], models: Dict[str, List[str]]) -> Dict[str, List[str]]:
    complete_dict = {}
    for dataset in datasets:
        complete_dict[dataset] = {}
        dataset_df = df[df['dataset_name'] == dataset]
        for model_family, model_variants in models.items():
            complete_dict[dataset][model_family] = {}
            for model_variant in model_variants:
                model_df = dataset_df[dataset_df['model'] == model_variant]
                # Sort the scores by id to ensure consistent ordering
                model_df = model_df.sort_values(by='id')
                # Extract WER scores as a list
                scores = model_df['WER'].tolist()
                complete_dict[dataset][model_family][model_variant] = scores
    return complete_dict


def get_bootstrap_p_value(diffs, n_iterations=10000, use_normal_approximation=True):
    """
    Calculates a two-sided p-value from a bootstrap distribution.
    """
    n_samples = len(diffs)
    # Fast vectorized resampling
    rng = np.random.default_rng(seed=42)
    random_indices = rng.integers(0, n_samples, size=(n_iterations, n_samples))
    resampled_diffs = diffs[random_indices]
    means = np.mean(resampled_diffs, axis=1)
    
    # Calculate fraction of times the mean crossed zero (or touched it)
    # If the observed diff is positive, we count how many bootstrap means were <= 0
    # If observed is negative, count how many >= 0
    observed_mean = np.mean(diffs)

    if use_normal_approximation:
        # Normal approximation for two-sided p-value
        std_error = np.std(diffs, ddof=1) / np.sqrt(n_samples)
        z_score = observed_mean / std_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return min(p_value, 1.0)
    
    if observed_mean > 0:
        tail_count = np.sum(means <= 0)
    else:
        tail_count = np.sum(means >= 0)
        
    # Two-sided p-value calculation
    # We add 1 to avoid p=0 (standard practice in permutation/bootstrap tests)
    p_value = (2 * tail_count + 1) / (n_iterations + 1)
    
    # Cap p-value at 1.0
    return min(p_value, 1.0)

def compare_model_family(model_dict, dataset: str, n_iterations=10000):
    """
    Args:
        model_dict: {'ModelA': [scores...], 'ModelB': [scores...], ...}
    """
    model_names = list(model_dict.keys())
    # Generate all 28 pairs from the 8 models
    pairs = list(itertools.combinations(model_names, 2))
    
    results = []
    
    print(f"Processing {len(pairs)} pairs with Bootstrap...")
    
    for m1, m2 in tqdm(pairs):
        # Calculate raw difference vector
        diffs = np.array(model_dict[m1]) - np.array(model_dict[m2])
        
        # Get raw p-value from bootstrap
        p_raw = get_bootstrap_p_value(diffs, n_iterations)
        
        results.append({
            'Model 1': m1,
            'Model 2': m2,
            'Mean Diff': np.mean(diffs),
            'p_raw': p_raw,
            'Dataset': dataset
        })
    
    df = pd.DataFrame(results)
    
    # --- The Correction Step ---
    # We use Holm-Bonferroni ('holm') correction
    reject, p_corrected, _, _ = multipletests(df['p_raw'], alpha=0.05, method='holm')
    
    df['p_corrected'] = p_corrected
    df['Significant'] = reject
    
    return df


def perform_pairwise_comparisons(
        scores_dict: Dict[str, Dict[str, List[str]]], 
        models: Dict[str, List[str]], 
        datasets: List[str],
        save_dir: Optional[Path]=None
    ) -> List[Path]:
    """
    Perform pairwise model comparisons for each dataset and model family.
    """
    paths = []
    for dataset in datasets:
        for model_family in models.keys():
            logger.info(f"Performing pairwise comparisons for {model_family} on {dataset}...")
            model_dict = scores_dict[dataset][model_family]
            result_df = compare_model_family(model_dict, dataset)
            
            logger.info(f"Saving results for {model_family} on {dataset}...")
            if save_dir is None:
                save_dir = Path("reports/statistical_analysis/")
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"bootstrap_comparison_{dataset}_{model_family.lower()}.parquet"
            result_df.to_parquet(save_dir / file_name, index=False)
            append_path = save_dir / file_name
            paths.append(append_path)
    return paths


def plot_pvalue_heatmap(
        parquet_file: str, 
        save_dir: Optional[str]=None, 
        format_dict: Optional[Dict]=None
    ) -> None:
    """
    Plots a heatmap of raw p-values (Holm-Bonferroni corrected).
    Linear scale: 0 = Dark Blue (Significant), 1 = White (Not Significant).
    """
    # 1. Load Data
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error loading {parquet_file}: {e}")
        return

    # 2. Extract Models
    models = sorted(list(set(df['Model 1']).union(set(df['Model 2']))))
    n_models = len(models)

    xticks = [_fmt(m, format_dict) for m in models]
    yticks = [_fmt(m, format_dict) for m in models]

    # 3. Create Matrices
    # p_matrix stores the actual p-values for both color and text
    p_matrix = pd.DataFrame(np.ones((n_models, n_models)), index=models, columns=models)

    for _, row in df.iterrows():
        m1, m2 = row['Model 1'], row['Model 2']
        p_val = row['p_corrected']
        # Fill Symmetric
        p_matrix.loc[m1, m2] = p_val
        p_matrix.loc[m2, m1] = p_val

    # 4. Create Annotation Matrix (Smart Text)
    annot_labels = p_matrix.astype(str).copy()
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                annot_labels.loc[m1, m2] = ""
                continue
            p_val = p_matrix.loc[m1, m2]
            # Smart Formatting
            if p_val < 0.001:
                txt = "<.001"
            else:
                txt = f"{p_val:.3f}"

            # If p is significant, add a "*"
            if p_val <= 0.05:
                txt += "*"
            
            annot_labels.loc[m1, m2] = txt

    # 5. Plotting
    plt.figure(figsize=(9, 7))
    base_name = os.path.basename(parquet_file).replace('.parquet', '')
    clean_title = base_name.replace('bootstrap_comparison_', '').replace('_', ' ').title()
    dataset = clean_title.split()[0]
    model_family = clean_title.split()[1]

    # Heatmap of RAW P-VALUES
    # vmin=0, vmax=1 ensure the full range is represented linearly
    # cmap="Blues_r" means 0 is Dark Blue, 1 is Light/White
    ax = sns.heatmap(p_matrix, annot=annot_labels, fmt="",
    cmap="viridis_r",
    vmin=0, vmax=1,
    linewidths=1, linecolor='white',
    xticklabels=xticks,
    yticklabels=yticks,
    cbar_kws={'label': 'Adjusted P-Value (Holm-Bonferroni)'})

    # 6. Format the text in the heatmap based on significance
    for text in ax.texts:
        try:
            val_txt = text.get_text()
            # We need the numerical value to decide styling
            # Handle the "<.001" case
            if "<" in val_txt or "*" in val_txt:
                val_float = 0.0
            else:
                val_float = float(val_txt)
            # Logic:
            # 1. If not significant (> 0.05), make text Grey and Normal weight
            # 2. If significant (<= 0.05), make text Black and Bold
            # 3. If extremely significant (<= 0.005), make text White (for contrast on dark blue)
            if val_float <= 0.05:
                text.set_fontweight('bold')

        except Exception as e:
            pass

    plt.title(f"Pairwise WER Comparison for {model_family}-models on {dataset}", fontsize=14)
    plt.ylabel("Model A")
    plt.xlabel("Model B")
    plt.tight_layout()

    # 7. Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base_name}_pvalues.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")
        plt.close()
    else:
        plt.close()



def pairwise_comparison_pipeline():
    """
    Full pipeline to perform pairwise comparisons and generate heatmaps.
    """
    results_path = Path("reports/metrics/")
    file_name = "combined_detailed_results_with_embeddings.parquet"
    logger.info(f"Loading data from {results_path / file_name}")
    df = load_from_parquet(results_path / file_name)
    logger.info(f"Successfully loaded data {len(df)} rows")

    # 1. Extract Datasets
    datasets = df['dataset_name'].unique().tolist()

    # 2. Build Scores Dictionary
    logger.info("Building scores dictionary...")
    scores_dict = build_scores_dict(df, datasets, MODELS)

    # 3. Perform Pairwise Comparisons
    logger.info("Performing pairwise comparisons...")
    result_paths = perform_pairwise_comparisons(scores_dict, MODELS, datasets, save_dir=Path("reports/statistical_analysis/"))

    # 4. Expand FORMAT_DICT for easier labeling
    expanded = expand_dict(FORMAT_DICT)

    # 5. Generate Heatmaps
    files_to_process = result_paths
    logger.info("Generating heatmaps for all pairwise comparison results...")

    for file in files_to_process:
        plot_pvalue_heatmap(file, format_dict=expanded, save_dir=Path("reports/statistical_analysis/"))