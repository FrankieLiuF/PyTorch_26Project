"""
Batch experiment runner for traditional ML baseline methods.

Loops over traditional classifiers (SVM, Random Forest, KNN, Decision Tree), 
runs 5-fold CV for each, and saves results incrementally to CSV.
Re-running automatically skips methods that already have results.

Configuration:
    Edit the variables at the top of main() to control datasets and methods.
"""

import sys
from pathlib import Path

# Solve loky (joblib) warning on Windows 11 where WMIC is deprecated.
# os.environ is inherited by subprocesses; filterwarnings covers the main process.
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from core.config import Config
from core.data_loader import load_dataset
from core.traditional import run_traditional_experiment


def main():
    # iris, parkinsons, hepatitis, acute_inflammations, zoo, hayes_roth
    DATASET = 'hayes_roth'
    # 5 traditional baselines for comparison with the image-based methods
    METHODS = ['svm', 'random_forest', 'knn', 'decision_tree', 'xgboost']

    print(f"\nDataset: {DATASET}  Methods: {METHODS}")
    print(f"Folds: {Config.N_FOLDS}  Seed: {Config.SEED}")

    # Same data and same StratifiedKFold seed as transfer learning
    X, y, feature_names, target_names = load_dataset(DATASET)

    Config.create_dirs()

    results_csv = Config.RESULTS_DIR / f'{DATASET}_traditional.csv'
    all_results = []
    done = set()

    # Load existing results to skip already-completed methods on re-run
    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        done = set(df_existing['model'].unique())
        # Load old rows so to_csv merges instead of overwrites
        all_results = df_existing.to_dict('records')
        print(f"Found {len(done)} completed method(s), will skip")

    total = len(METHODS)
    print(f"Total: {total}  Done: {len(done)}\n")

    count = 0

    for method in METHODS:
        count += 1

        if method in done:
            print(f"[{count}/{total}] {method}  skip")
            continue

        print(f"\n[{count}/{total}] {method}")
        try:
            # run_traditional_experiment handles 5-fold CV internally
            # and returns all 5 folds at once
            fold_results = run_traditional_experiment(
                X, y,
                model_name=method,
                n_folds=Config.N_FOLDS,
                seed=Config.SEED
            )

            for r in fold_results:
                all_results.append({
                    'fold': r['fold'],
                    'method': 'traditional',
                    'model': r['model'],
                    'accuracy': round(r['accuracy'], 4),
                    'f1_score': round(r['f1_score'], 4),
                    # Preserve NaN when AUC is undefined (e.g. missing classes in a small fold)
            'auc_roc': round(r['auc_roc'], 4) if not np.isnan(r['auc_roc']) else float('nan'),
                })

        except Exception as e:
            print(f"ERROR: {e}")

        # Save after each method (traditional is fast, so per-method is fine)
        pd.DataFrame(all_results).to_csv(results_csv, index=False, na_rep='NaN')

    print(f"\nDone: {results_csv}")

    df = pd.read_csv(results_csv)
    print("\nMean ± std per method:")
    summary = df.groupby('model')[['accuracy', 'f1_score', 'auc_roc']].agg(['mean', 'std']).round(4)
    print(summary)


if __name__ == '__main__':
    main()
