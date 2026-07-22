"""
Batch experiment runner for transfer learning with TINTOlib image methods.

Loops over transformation methods × CNN models × cross-validation folds,
trains each combination, and saves results incrementally to CSV.
Re-running automatically skips combinations that already have results.

Usage:
    python experiments/run_transfer.py

Configuration:
    Edit the variables at the top of main() to control datasets, methods,
    models, and training parameters.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so core/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.config import Config
from core.data_loader import load_dataset
from core.trainer import train_transfer_learning


def main():
    # ============================================================
    # Experiment configuration — edit here to change what runs
    # ============================================================
    DATASET = 'iris'
    METHODS = ['tinto']
    MODELS = ['efficientnet_v2_m', 'mobilenet_v3_large']
    EPOCHS = 50
    DEVICE = Config.get_device()

    # ============================================================
    # 1. Load data
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Dataset: {DATASET}")
    print(f"Methods: {METHODS}")
    print(f"Models : {MODELS}")
    print(f"Folds  : {Config.N_FOLDS}")
    print(f"Epochs : {EPOCHS}")
    print(f"Device : {DEVICE}")
    print(f"{'='*60}")

    X, y, feature_names, target_names = load_dataset(DATASET)
    Config.NUM_CLASSES = len(target_names)  # override iris default for this dataset

    Config.create_dirs()

    # ============================================================
    # 2. Load existing results and build skip-set
    # ============================================================
    results_csv = Config.RESULTS_DIR / f'{DATASET}_transfer.csv'
    done = set()

    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        done = set(zip(df_existing['fold'], df_existing['method'], df_existing['model']))
        print(f"\nFound {len(done)} existing result(s) — will skip")

    # ============================================================
    # 3. Triple loop: method × model × fold
    # ============================================================
    total = len(METHODS) * len(MODELS) * Config.N_FOLDS
    print(f"Total combinations: {total}  |  Already done: {len(done)}\n")

    results = []
    count = 0

    for method in METHODS:
        for model in MODELS:
            for fold in range(Config.N_FOLDS):

                if (fold, method, model) in done:
                    print(f"[{count+1}/{total}] fold={fold} method={method} model={model} → skip")
                    count += 1
                    continue

                # --- Train single combination ---
                print(f"\n[{count+1}/{total}] fold={fold} method={method} model={model}")
                try:
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

                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        'fold': fold,
                        'method': method,
                        'model': model,
                        'accuracy': None,
                        'f1_score': None,
                        'auc_roc': None,
                    })

                # Write after every fold so a crash preserves earlier work
                pd.DataFrame(results).to_csv(results_csv, index=False)
                count += 1

    # ============================================================
    # 4. Print summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {results_csv}")
    print(f"{'='*60}")

    df = pd.read_csv(results_csv)
    print("\nSummary (mean ± std per method × model):")
    summary = df.groupby(['method', 'model'])[['accuracy', 'f1_score', 'auc_roc']].agg(['mean', 'std']).round(4)
    print(summary)


if __name__ == '__main__':
    main()
