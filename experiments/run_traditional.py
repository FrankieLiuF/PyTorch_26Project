"""
Batch experiment runner for traditional ML baseline methods.

Loops over traditional classifiers (SVM, Random Forest, KNN, Decision Tree,
XGBoost), runs 5-fold CV for each, and saves results incrementally to CSV.
Re-running automatically skips methods that already have results.

Usage:
    python experiments/run_traditional.py

Configuration:
    Edit the variables at the top of main() to control datasets and methods.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so core/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from core.config import Config
from core.data_loader import load_dataset
from core.traditional import run_traditional_experiment


def main():
    # ============================================================
    # Experiment configuration — edit here to change what runs
    # ============================================================
    DATASET = 'iris'
    METHODS = ['svm', 'random_forest', 'knn', 'decision_tree', 'xgboost']

    # ============================================================
    # 1. Load data
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Dataset : {DATASET}")
    print(f"Methods : {METHODS}")
    print(f"Folds   : {Config.N_FOLDS}")
    print(f"Seed    : {Config.SEED}")
    print(f"{'='*60}")

    X, y, feature_names, target_names = load_dataset(DATASET)

    Config.create_dirs()

    # ============================================================
    # 2. Load existing results and build skip-set
    # ============================================================
    results_csv = Config.RESULTS_DIR / f'{DATASET}_traditional.csv'
    done = set()

    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        done = set(df_existing['model'].unique())  # 'model' column = method name
        print(f"\nFound {len(done)} completed method(s) — will skip")

    # ============================================================
    # 3. Loop over traditional methods
    # ============================================================
    total = len(METHODS)
    print(f"Total methods: {total}  |  Already done: {len(done)}\n")

    all_results = []
    count = 0

    for method in METHODS:
        count += 1

        if method in done:
            print(f"[{count}/{total}] {method} → skip")
            continue

        # --- Run 5-fold CV for one method ---
        print(f"\n[{count}/{total}] {method}")
        try:
            fold_results = run_traditional_experiment(
                X, y,
                model_name=method,
                n_folds=Config.N_FOLDS,
                seed=Config.SEED
            )

            # Only keep columns that match transfer results CSV schema
            for r in fold_results:
                all_results.append({
                    'fold': r['fold'],
                    'method': 'traditional',
                    'model': r['model'],
                    'accuracy': round(r['accuracy'], 4),
                    'f1_score': round(r['f1_score'], 4),
                    'auc_roc': round(r['auc_roc'], 4) if not np.isnan(r['auc_roc']) else None,
                })

        except Exception as e:
            print(f"  ERROR: {e}")

        # Write after every method so a crash preserves earlier work
        pd.DataFrame(all_results).to_csv(results_csv, index=False)

    # ============================================================
    # 4. Print summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {results_csv}")
    print(f"{'='*60}")

    df = pd.read_csv(results_csv)
    print("\nSummary (mean ± std per method):")
    summary = df.groupby('model')[['accuracy', 'f1_score', 'auc_roc']].agg(['mean', 'std']).round(4)
    print(summary)


if __name__ == '__main__':
    main()
