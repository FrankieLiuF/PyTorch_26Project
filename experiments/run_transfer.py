"""
Batch experiment runner for transfer learning with TINTOlib image methods.

Loops over transformation methods × CNN models × cross-validation folds,
trains each combination, and saves results incrementally to CSV.
Re-running automatically skips combinations that already have results.

Configuration:
    Edit the variables at the top of main() to control datasets, methods,
    models, and training parameters.
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

import pandas as pd
from core.config import Config
from core.data_loader import load_dataset
from core.trainer import train_transfer_learning


def main():
    # iris, parkinsons, hepatitis, acute_inflammations, zoo, hayes_roth
    DATASET = 'parkinsons'
    # Full run: 5 methods × 4 models × 5 folds = 100 runs (10 already done → 90 new)
    # 'tinto', 'igtd', 'supertml', 'refined', 'deepinsight'
    METHODS = ['tinto', 'igtd', 'supertml', 'refined', 'deepinsight']
    # 'efficientnet_v2_m', 'mobilenet_v3_large', 'resnext50_32x4d', 'densenet161'
    MODELS = ['efficientnet_v2_m', 'mobilenet_v3_large', 'resnext50_32x4d', 'densenet161']
    EPOCHS = 50
    DEVICE = Config.get_device()

    print(f"\nDataset: {DATASET}  Methods: {METHODS}  Models: {MODELS}")
    print(f"Folds: {Config.N_FOLDS}  Epochs: {EPOCHS}  Device: {DEVICE}")

    # Load tabular data from UCI dataset (preprocessed CSV + JSON)
    X, y, feature_names, target_names = load_dataset(DATASET)
    Config.NUM_CLASSES = len(target_names)

    Config.create_dirs()

    results_csv = Config.RESULTS_DIR / f'{DATASET}_transfer.csv'

    # Load existing results so re-runs only fill gaps and CSV is preserved
    results = []
    done = set()

    # Build set of completed (fold, method, model) tuples to skip on re-run
    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        done = set(zip(df_existing['fold'], df_existing['method'], df_existing['model']))
        # Load old rows into results so to_csv merges instead of overwrites
        results = df_existing.to_dict('records')
        print(f"Found {len(done)} existing result(s), will skip")

    total = len(METHODS) * len(MODELS) * Config.N_FOLDS
    print(f"Total: {total}  Done: {len(done)}\n")

    count = 0

    # Triple nested loop: each method × model × fold combination is one independent run
    for method in METHODS:
        for model in MODELS:
            for fold in range(Config.N_FOLDS):

                if (fold, method, model) in done:
                    print(f"[{count+1}/{total}] fold={fold} {method} {model}  skip")
                    count += 1
                    continue

                print(f"[{count+1}/{total}] fold={fold} {method} {model}")
                try:
                    # Core call: generate images -> train CNN classifier -> return metrics
                    _, _, metrics = train_transfer_learning(
                        X, y,
                        fold_idx=fold,
                        epochs=EPOCHS,
                        method_name=method,
                        model_name=model,
                        feature_names=feature_names,
                        device=DEVICE,
                    )

                    results.append({
                        'fold': fold,
                        'method': method,
                        'model': model,
                        'accuracy': round(metrics['accuracy'], 4),
                        'f1_score': round(metrics['f1_score'], 4),
                        'auc_roc': round(metrics['auc_roc'], 4),
                    })

                # On error, record NaN placeholders so the next combination can still run
                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append({
                        'fold': fold,
                        'method': method,
                        'model': model,
                        'accuracy': None,
                        'f1_score': None,
                        'auc_roc': None,
                    })

                # Save after each fold so a crash preserves completed work
                pd.DataFrame(results).to_csv(results_csv, index=False, na_rep='NaN')
                count += 1

    print(f"\nDone: {results_csv}")

    df = pd.read_csv(results_csv)
    print("\nMean +- std per method x model:")
    # the right order: groupby
    summary = df.groupby(['method', 'model'])[['accuracy', 'f1_score', 'auc_roc']].agg(['mean', 'std']).round(4)
    print(summary)


if __name__ == '__main__':
    main()
