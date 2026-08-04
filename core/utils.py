"""
08.Utility functions for visualization, results saving, and statistical tests

Functions:
    1. plot_training_curves
    2. save_results_to_csv
    3. load_results_from_csv
    4. perform_wilcoxon_tests
    5. print_wilcoxon_summary
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations
from pathlib import Path


# Visualization functions
def plot_training_curves(history, save_path=None):
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        history: Training history from train_transfer_learning 
        save_path: Path to save the figure 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax1.plot(history['val_acc'], label='Val Accuracy', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0, 1.05])

    # Loss plot
    ax2.plot(history['train_loss'], label='Train Loss', marker='o')
    ax2.plot(history['val_loss'], label='Val Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curves saved to: {save_path}")

    plt.show()


# Results Saving functions
def save_results_to_csv(all_results, output_path):
    """
    Save all experiment results to CSV.
    
    Args:
        all_results: List of result dictionaries 
        output_path: Output file path 
    
    Returns:
        DataFrame: Saved results
    """
    # Transfer results to pandas DataFrame
    df = pd.DataFrame(all_results)

    # Make sure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as csv, do not saving index
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


# Load results from CSV
def load_results_from_csv(filepath):
    """Load experiment results from CSV"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} results from: {filepath}")
    return df


# Wilcoxon Signed-Rank Test Functions
def perform_wilcoxon_tests(df, metric='accuracy', group_by='method'):
    """
    Perform Wilcoxon Signed-Rank Test between all pairs of methods/models.
    
    Args:
        df: DataFrame containing experiment results 
        metric: Metric to compare ('accuracy', 'f1_score', etc.) 
        group_by: Column name for grouping ('method' or 'model') 
    
    Returns:
        DataFrame: Test results for all pairs 
    """
    # Get all groups that need to be compared
    groups = df[group_by].unique()

    results = []

    # combinations is the itertools, generate all possible combinations of length of 2
    for g1, g2 in combinations(groups, 2):
        df1 = df[df[group_by] == g1].sort_values('fold')
        df2 = df[df[group_by] == g2].sort_values('fold')

        scores1 = df1[metric].values
        scores2 = df2[metric].values

        if len(scores1) != len(scores2):
            continue

        # Check if all fold index matches
        if not (df1['fold'].values == df2['fold'].values).all():
            continue
        
        try:
            statistic, p_value = wilcoxon(scores1, scores2)
        except ValueError as e:
            continue

        mean_diff = np.mean(scores1) - np.mean(scores2)

        results.append({
            'group1': g1, 
            'group2': g2,
            'mean1': np.mean(scores1), 
            'mean2': np.mean(scores2),
            'mean_diff': mean_diff,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better': g1 if mean_diff > 0 else g2
        })

    return pd.DataFrame(results)


# Print formatted Wilcoxon test results in a readable format
def print_wilcoxon_summary(results_df):
    print("\n" + "="*70)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("-" * 70)
    print("p < 0.05 indicates significant difference")
    print("="*70)

    # Sort by p ascendingly
    sorted_df = results_df.sort_values('p_value')

    for _, row in results_df.sort_values('p_value').iterrows():
        sig = "Significant" if row['significant'] else "Not significant"

        if row['mean_diff'] > 0:
            better = f"{row['group1']} (higher)"
        elif row['mean_diff'] < 0:
            better = f"{row['group2']} (higher)"
        else:
            better = "Tie"

        print(f"\n{row['group1']} vs {row['group2']}:")
        print(f"  Mean1: {row['mean1']:.4f}, Mean2: {row['mean2']:.4f}")
        print(f"  Mean difference: {row['mean_diff']:.4f} -> {better}")
        print(f"  p-value: {row['p_value']:.6f} → {sig}")


