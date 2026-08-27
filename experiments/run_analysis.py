"""
Aggregate transfer and traditional experiment results into summary tables
and statistical tests (Friedman + Nemenyi, Wilcoxon).

Functions:
    1. load_all_results  — load all transfer and traditional CSVs into one DataFrame
    2. compute_summary   — mean +- std per method per dataset
    3. compute_friedman  — Friedman + Nemenyi ranking and letter groups
    4. compute_wilcoxon  — best transfer vs best traditional per dataset

Usage:
    conda activate CHM9360
    python experiments/run_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.config import Config
from core.utils import friedman_nemenyi_test, compare_best_methods, save_results_to_csv

DATASETS = ['iris', 'parkinsons', 'hepatitis', 'acute_inflammations', 'zoo', 'hayes_roth']
METRICS = ['accuracy', 'f1_score', 'auc_roc']
TRANSFER = ['tinto', 'igtd', 'supertml', 'refined', 'deepinsight']
TRADITIONAL = ['svm', 'random_forest', 'knn', 'decision_tree', 'xgboost']


def load_all_results():
    """
    Load all transfer and traditional CSVs, merge into one unified DataFrame.

    Each transfer method+model combination is kept as a separate entry
    (e.g. tinto+efficientnet_v2_m). Traditional results have the
    'model' column renamed to 'method' for a consistent schema.

    Returns:
        DataFrame with columns: dataset, fold, method, accuracy, f1_score, auc_roc
    """
    all_rows = []

    # Loop over each dataset and load its transfer + traditional CSV
    for ds in DATASETS:
        # Transfer: combine method and model name into one label
        path_t = Config.RESULTS_DIR / f'{ds}_transfer.csv'
        if path_t.exists():
            df = pd.read_csv(path_t)
            df['method'] = df['method'] + '+' + df['model']
            df = df[['fold', 'method'] + METRICS]
            df['dataset'] = ds
            all_rows.append(df)

        # Traditional: the 'model' column is the method (svm, random_forest, ...)
        path_tr = Config.RESULTS_DIR / f'{ds}_traditional.csv'
        if path_tr.exists():
            df = pd.read_csv(path_tr)
            df = df[['fold', 'model'] + METRICS].rename(columns={'model': 'method'})
            df['dataset'] = ds
            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def compute_summary(df):
    """
    Compute mean +- std for every method, per dataset.

    Returns:
        DataFrame with columns: dataset, method, accuracy_mean, accuracy_std,
        f1_score_mean, f1_score_std, auc_roc_mean, auc_roc_std.
    """
    print("=" * 80)
    print("PER-DATASET SUMMARY (mean +- std over 5 folds)")
    print("=" * 80)

    all_agg = []

    for ds in DATASETS:
        df_ds = df[df['dataset'] == ds]

        # Aggregate mean and std for each method across the 5 folds
        agg = df_ds.groupby('method')[METRICS].agg(['mean', 'std']).round(4)
        agg.columns = [f'{c[0]}_{c[1]}' for c in agg.columns]
        agg = agg.sort_values('accuracy_mean', ascending=False)
        agg['dataset'] = ds
        agg['method'] = agg.index
        all_agg.append(agg)

        # Print formatted table for this dataset
        print(f"\n{'─' * 70}")
        print(f"  {ds.upper()}  ({len(df_ds['fold'].unique())} folds)")
        print(f"{'─' * 70}")
        hdr = f"{'Method':<20} {'Accuracy':>14} {'F1-Score':>14} {'AUC-ROC':>14}"
        print(hdr)
        print('-' * len(hdr))

        for method, row in agg.iterrows():
            a = f"{row['accuracy_mean']:.4f} +- {row['accuracy_std']:.4f}"
            f = f"{row['f1_score_mean']:.4f} +- {row['f1_score_std']:.4f}"
            u = f"{row['auc_roc_mean']:.4f} +- {row['auc_roc_std']:.4f}"
            print(f"{method:<20} {a:>14} {f:>14} {u:>14}")

    # Combine all datasets, reorder columns
    cols = ['dataset', 'method'] + [f'{m}_{s}' for m in METRICS for s in ['mean', 'std']]
    return pd.concat(all_agg, ignore_index=True)[cols]


def compute_friedman(df, metric='accuracy'):
    """
    Run Friedman + Nemenyi per dataset.

    Returns:
        DataFrame with columns: dataset, method, friedman_stat, friedman_p,
        significant, avg_rank, group.
    """
    print("\n" + "=" * 80)
    print(f"FRIEDMAN + NEMENYI ({metric})")
    print("=" * 80)

    rows = []

    for ds in DATASETS:
        df_ds = df[df['dataset'] == ds]

        # Pivot to (5 folds) x (9 methods) — each cell is one accuracy score
        pivot = df_ds.pivot_table(index='fold', columns='method', values=metric)
        r = friedman_nemenyi_test(pivot)

        print(f"\n{'─' * 70}")
        sig = "***" if r['significant'] else "n.s."
        print(f"  {ds.upper()}  "
              f"(n={r['n_folds']} folds, k={r['n_methods']} methods)")
        print(f"  Friedman stat = {r['friedman_stat']:.4f}, "
              f"p = {r['friedman_p']:.6f}  {sig}")
        print(f"{'─' * 70}")

        # Friedman not significant — still record each method, mark as n.s.
        if not r['significant']:
            print("  No significant difference among methods.\n")
            for method in pivot.columns:
                rows.append({
                    'dataset': ds, 'method': method, 'metric': metric,
                    'friedman_stat': r['friedman_stat'],
                    'friedman_p': r['friedman_p'],
                    'significant': False,
                    'avg_rank': None, 'group': 'n.s.',
                })
            continue

        # Significant: print Nemenyi ranking with letter groups
        print(f"  {'Method':<20} {'Avg Rank':>10}  Group")
        print(f"  {'─'*40}")
        for method, rank in r['avg_ranks'].items():
            print(f"  {method:<20} {rank:>8.2f}   {r['groups'][method]}")
            rows.append({
                'dataset': ds, 'method': method, 'metric': metric,
                'friedman_stat': r['friedman_stat'],
                'friedman_p': r['friedman_p'],
                'significant': True,
                'avg_rank': round(rank, 2),
                'group': r['groups'][method],
            })

    return pd.DataFrame(rows)


def compute_wilcoxon(df, metric='accuracy'):
    """
    Compare best transfer vs best traditional per dataset using Wilcoxon.

    Returns:
        DataFrame with columns: dataset, best_transfer, best_traditional,
        mean_transfer, mean_traditional, wilcoxon_p.
    """
    print("\n" + "=" * 80)
    print(f"WILCOXON: Best Transfer vs Best Traditional ({metric})")
    print("=" * 80)

    # Transfer methods contain '+' (e.g. tinto+efficientnet_v2_m)
    transfer_names = [m for m in df['method'].unique() if '+' in m]

    rows = []

    for ds in DATASETS:
        df_ds = df[df['dataset'] == ds]
        r = compare_best_methods(df_ds, transfer_names, TRADITIONAL, metric)

        if r is None:
            print(f"\n  {ds}: SKIP (missing data)")
            continue

        p = r['wilcoxon_p']
        # Significance stars: *** p<0.01, ** p<0.05, * p<0.10
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        print(f"\n  {ds}: {r['best_a']} ({r['mean_a']:.4f}) vs "
              f"{r['best_b']} ({r['mean_b']:.4f})")
        print(f"    Wilcoxon p = {p:.6f} {sig}")

        rows.append({
            'dataset': ds, 'metric': metric,
            'best_transfer': r['best_a'], 'best_traditional': r['best_b'],
            'mean_transfer': round(r['mean_a'], 4),
            'mean_traditional': round(r['mean_b'], 4),
            'wilcoxon_p': round(p, 6),
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load_all_results()
    print(f"Loaded {len(df)} rows  "
          f"Datasets: {df['dataset'].nunique()}  Methods: {df['method'].nunique()}")
    print(f"Methods: {sorted(df['method'].unique())}")

    # Sanity check: each (dataset, method) should have exactly 5 folds
    counts = df.groupby(['dataset', 'method']).size()
    incomplete = counts[counts != 5]
    if len(incomplete) > 0:
        print(f"\nWARNING - incomplete entries:\n{incomplete}")
    else:
        print("All (dataset, method) have exactly 5 folds.\n")

    # Compute and print all three analysis tables
    df_summary = compute_summary(df)
    df_friedman = compute_friedman(df)
    df_wilcoxon = compute_wilcoxon(df)

    # Save results to CSV for easy lookup and paper table preparation
    out_dir = Config.RESULTS_DIR
    save_results_to_csv(df_summary.to_dict('records'), out_dir / 'analysis_summary.csv',
                        float_format='%.4f')
    save_results_to_csv(df_friedman.to_dict('records'), out_dir / 'analysis_friedman.csv',
                        float_format='%.4f')
    save_results_to_csv(df_wilcoxon.to_dict('records'), out_dir / 'analysis_wilcoxon.csv',
                        float_format='%.6f')
