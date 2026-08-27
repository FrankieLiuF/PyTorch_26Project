"""
Save training history (loss curves) for representative transfer runs.

Unlike run_transfer.py, this script runs only a hand-picked list of
(dataset, method, model, fold) combinations, keeps the full per-epoch
history, saves it as pickle, and plots train/val loss + accuracy curves.

"""

import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.data_loader import load_dataset
from core.trainer import train_transfer_learning
from core.utils import plot_training_curves

# Representative runs to record. Pick one good combo per dataset. 
# (The results are picked from the final analysis summary)
RUNS = [
    ('iris',                  'supertml',   'resnext50_32x4d',  0),
    ('parkinsons',            'igtd',       'densenet161',      1),
    ('hepatitis',             'deepinsight', 'mobilenet_v3_large', 0),
    ('acute_inflammations',   'tinto',      'densenet161',      0),
    ('zoo',                   'refined',    'resnext50_32x4d',  0),
    ('hayes_roth',            'supertml',   'densenet161',      2),
]

EPOCHS = 50


def main():
    device = Config.get_device()
    out_dir = Config.RESULTS_DIR / 'history'
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, method, model, fold in RUNS:
        print(f"\nDataset: {dataset}  method: {method}  model: {model}  fold: {fold}")

        X, y, feature_names, target_names = load_dataset(dataset)
        Config.NUM_CLASSES = len(target_names)

        # Keep the full history this time (skip trained model)
        history, _, metrics = train_transfer_learning(
            X, y,
            fold_idx=fold,
            epochs=EPOCHS,
            method_name=method,
            model_name=model,
            feature_names=feature_names,
            device=device,
        )

        tag = f'{dataset}_{method}_{model}_fold{fold}'

        # Save raw history for later re-plotting
        pkl_path = out_dir / f'{tag}.pkl'

        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)
            
        print(f"History saved: {pkl_path}")

        # Plot train/val loss and accuracy curves.
        # Note: if plt.show() is enabled in plot_training_curves, the script
        # blocks at each run until the figure window is closed manually.
        fig_path = out_dir / f'{tag}.png'
        plot_training_curves(history, save_path=fig_path)

        print(f"Metrics: acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1_score']:.4f} auc={metrics['auc_roc']:.4f}")


if __name__ == '__main__':
    main()
