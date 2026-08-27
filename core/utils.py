"""
08.Utility functions for visualization, results saving, and statistical tests

Functions:
    1. plot_training_curves
    2. save_results_to_csv
    3. load_results_from_csv
    4. perform_wilcoxon_tests
    5. print_wilcoxon_summary
    6. friedman_nemenyi_test
    7. _assign_letter_groups
    8. compare_best_methods
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations
from pathlib import Path
import scikit_posthocs as sp


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

    plt.show()  # commented out for batch runs; uncomment to view interactively


def save_results_to_csv(all_results, output_path, float_format=None):
    """
    Save all experiment results to CSV.

    Args:
        all_results: List of result dictionaries
        output_path: Output file path
        float_format: Format string for floats (e.g. '%.4f' keeps trailing zeros)

    Returns:
        DataFrame: Saved results
    """
    # Transfer results to pandas DataFrame
    df = pd.DataFrame(all_results)

    # Make sure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as csv, do not saving index
    df.to_csv(output_path, index=False, float_format=float_format)
    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def load_results_from_csv(filepath):
    """Load experiment results from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} results from: {filepath}")
    return df


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
        except ValueError:
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
            'better': g1 if mean_diff > 0 else g2,
        })

    return pd.DataFrame(results)


def print_wilcoxon_summary(results_df):
    """Print formatted Wilcoxon test results in a readable format."""
    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("-" * 70)
    print("p < 0.05 indicates significant difference")
    print("=" * 70)

    # Sort by p ascendingly
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
        print(f"  p-value: {row['p_value']:.6f} -> {sig}")


def friedman_nemenyi_test(pivot):
    """
    Run Friedman test + Nemenyi post-hoc on a pivot table of fold scores.

    Args:
        pivot: DataFrame with rows=folds, columns=methods, values=scores

    Returns:
        dict with keys friedman_stat, friedman_p, significant,
        n_methods, n_folds, avg_ranks, groups, p_matrix.
        avg_ranks/groups/p_matrix are None if Friedman not significant.
    """
    # Drop methods that have NaN (e.g. zoo fold 1 auc_roc)
    pivot_clean = pivot.dropna(axis=1)
    methods = list(pivot_clean.columns)

    # Friedman test — one array of scores per method
    scores_per_method = [pivot_clean[m].values for m in methods]
    # *: unpacking the elements (each array in the list)
    stat, p = friedmanchisquare(*scores_per_method)

    result = {
        'friedman_stat': stat,
        'friedman_p': p,
        'significant': p < 0.05,
        'n_methods': len(methods),
        'n_folds': len(pivot_clean),
        'avg_ranks': None,
        'groups': None,
        'p_matrix': None,
    }

    # No significant difference — skip post-hoc, return early
    if p >= 0.05:
        return result

    # Nemenyi post-hoc: expects (blocks, treatments) = (folds, methods)
    p_matrix = sp.posthoc_nemenyi_friedman(pivot_clean)

    # Average rank per method within each fold (1 = best), then mean across folds
    ranks = pivot_clean.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Convert p-value matrix to compact letter display
    groups = _assign_letter_groups(p_matrix, avg_ranks)

    result['p_matrix'] = p_matrix
    result['avg_ranks'] = avg_ranks
    result['groups'] = groups
    return result


def _assign_letter_groups(p_matrix, avg_ranks):
    """
    Convert a Nemenyi p-value matrix into compact letter groups.

    Two methods share a letter if their pairwise p >= 0.05
    (not significantly different). Uses a greedy algorithm:
    iterate methods from best to worst rank; for each, join all
    existing letter groups whose members are all not-significantly-different
    from this method. If none, create a new letter.

    Args:
        p_matrix: DataFrame (methods x methods) of Nemenyi p-values
        avg_ranks: Series method -> average rank, sorted ascending

    Returns:
        dict: method -> compact letter string (e.g. 'a', 'a b', 'a-c')
    """
    methods = list(avg_ranks.index)
    method_letters = {}
    letter_members = []

    # Iterate methods from best to worst rank, greedily assign letters
    for method in methods:
        # Check which existing letter groups this method can join
        my_letters = set()
        for letter, members in letter_members:
            # Can join if NOT significantly different from ALL current members
            can_join = all(
                p_matrix.loc[method, m] >= 0.05
                for m in members
            )
            if can_join:
                my_letters.add(letter)

        # Cannot join any existing group, create a new letter
        if not my_letters:
            new_letter = chr(ord('a') + len(letter_members))
            my_letters = {new_letter}
            letter_members.append((new_letter, []))

        # Register this method into every group it belongs to
        for letter, members in letter_members:
            if letter in my_letters:
                members.append(method)

        method_letters[method] = my_letters

    return {m: ' '.join(sorted(letters)) for m, letters in method_letters.items()}


def compare_best_methods(df, group_a, group_b, metric='accuracy'):
    """
    Pick the best method (by mean metric) from each of two groups,
    then run Wilcoxon signed-rank test on their paired fold scores.

    Args:
        df: DataFrame with columns [fold, method, metric, ...]
        group_a: list of method names for group A
        group_b: list of method names for group B
        metric: metric column name

    Returns:
        dict with best_a, best_b, mean_a, mean_b, wilcoxon_stat, wilcoxon_p,
        or None if either group is empty.
    """
    df_a = df[df['method'].isin(group_a)]
    df_b = df[df['method'].isin(group_b)]

    if df_a.empty or df_b.empty:
        return None

    # Pick the method with highest mean score in each group
    best_a = df_a.groupby('method')[metric].mean().idxmax()
    best_b = df_b.groupby('method')[metric].mean().idxmax()

    # Extract the 5 paired fold scores for the two best methods
    scores_a = df_a[df_a['method'] == best_a].sort_values('fold')[metric].values
    scores_b = df_b[df_b['method'] == best_b].sort_values('fold')[metric].values

    # Wilcoxon signed-rank test on the paired fold scores
    try:
        stat, p = wilcoxon(scores_a, scores_b)
    except ValueError:
        stat, p = np.nan, np.nan

    return {
        'best_a': best_a,
        'best_b': best_b,
        'mean_a': scores_a.mean(),
        'mean_b': scores_b.mean(),
        'wilcoxon_stat': stat,
        'wilcoxon_p': p,
    }
