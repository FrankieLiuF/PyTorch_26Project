"""
Ablation runner for the image normalization choice.

Same grid as run_transfer.py (methods x models x folds) but trains with
use_simple_norm=True, i.e. mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] instead
of the ImageNet statistics. Results go to a separate CSV
({dataset}_transfer_simplenorm.csv) so the main results stay untouched.

At the end it prints a comparison against the ImageNet-normalized CSV
if it exists: per-combination delta and overall win/loss/tie counts.

Configuration:
    Edit DATASET at the top of main() to choose the dataset.
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


def compare_normalizations(dataset):
    """Compare simplenorm vs ImageNet CSV on identical (fold, method, model) combos."""
    main_csv = Config.RESULTS_DIR / f'{dataset}_transfer.csv'
    simple_csv = Config.RESULTS_DIR / f'{dataset}_transfer_simplenorm.csv'

    if not main_csv.exists() or not simple_csv.exists():
        print("Both CSVs are required for comparison, skip")
        return

    df_main = pd.read_csv(main_csv)
    df_simple = pd.read_csv(simple_csv)

    # Same fold+method+model rows must exist in both files to be comparable
    merged = df_main.merge(
        df_simple, on=['fold', 'method', 'model'],
        suffixes=('_imagenet', '_simple')
    )
    merged['acc_delta'] = merged['accuracy_simple'] - merged['accuracy_imagenet']

    wins = (merged['acc_delta'] > 0).sum()
    ties = (merged['acc_delta'] == 0).sum()
    losses = (merged['acc_delta'] < 0).sum()

    print("\n" + "=" * 70)
    print(f"COMPARISON: simple(0.5) vs ImageNet normalization ({dataset})")
    print("=" * 70)
    print(f"Combos compared: {len(merged)}")
    print(f"simple better: {wins}  tie: {ties}  simple worse: {losses}")
    print(f"Mean accuracy delta (simple - imagenet): {merged['acc_delta'].mean():+.4f}")


def main():
    # iris, parkinsons, hepatitis, acute_inflammations, zoo, hayes_roth
    DATASET = 'parkinsons'
    # 'tinto', 'igtd', 'supertml', 'refined', 'deepinsight'
    METHODS = ['tinto', 'igtd', 'supertml', 'refined', 'deepinsight']
    # 'efficientnet_v2_m', 'mobilenet_v3_large', 'resnext50_32x4d', 'densenet161'
    MODELS = ['efficientnet_v2_m', 'mobilenet_v3_large', 'resnext50_32x4d', 'densenet161']
    EPOCHS = 50
    DEVICE = Config.get_device()

    print(f"\nDataset: {DATASET}  Methods: {METHODS}  Models: {MODELS}")
    print(f"Folds: {Config.N_FOLDS}  Epochs: {EPOCHS}  Device: {DEVICE}")
    print("Normalization: simple (mean=0.5, std=0.5)")

    # Load tabular data from UCI dataset (preprocessed CSV + JSON)
    X, y, feature_names, target_names = load_dataset(DATASET)
    Config.NUM_CLASSES = len(target_names)

    Config.create_dirs()

    # Separate CSV so the main ImageNet-normalized results are never touched
    results_csv = Config.RESULTS_DIR / f'{DATASET}_transfer_simplenorm.csv'

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
                    # Same training call as run_transfer.py, only the
                    # normalization of the loaded images changes
                    _, _, metrics = train_transfer_learning(
                        X, y,
                        fold_idx=fold,
                        epochs=EPOCHS,
                        method_name=method,
                        model_name=model,
                        feature_names=feature_names,
                        device=DEVICE,
                        use_simple_norm=True,
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
    print("\nMean +- std per method x model (simple norm):")
    summary = df.groupby(['method', 'model'])[['accuracy', 'f1_score', 'auc_roc']].agg(['mean', 'std']).round(4)
    print(summary)

    compare_normalizations(DATASET)


if __name__ == '__main__':
    main()
