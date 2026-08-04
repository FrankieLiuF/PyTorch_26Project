"""
Traditional machine learning methods for baseline comparison

Functions:
    1. run_traditional_experiment
    2. run_all_traditional_methods
    3. get_traditional_summary
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

from .config import Config

# try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost will be skipped.")
    print("tip: pip install xgboost")


# Run 5-fold cross-validation for a single traditional ML method
def run_traditional_experiment(X, y, 
                               model_name='svm', 
                               n_folds=5, 
                               seed=42):
    """
    Run 5-fold cross-validation for a single traditional method.
    
    Args:
        X: Feature array 
        y: Label array 
        model_name: Name of the model 
        n_folds: Number of folds 
        seed: Random seed
    
    Returns:
        list: Results for each fold 
    """
    # Register of scikit-learn classifiers with fixed parameters for fair comparison
    models = {
        'svm': SVC(kernel='rbf', probability=True, random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=seed),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(random_state=seed),
    }

    if XGB_AVAILABLE:
        models['xgboost'] = XGBClassifier(
            n_estimators=100, 
            random_state=seed, 
            eval_metric='mlogloss'
        )

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    model = models[model_name]

    # Use same stratifiedKFold as TINTO methods
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize (consistent with TINTO pipeline)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        # Binary: positive-class probability. Multi-class: one-vs-rest AUROC
        # AUC-ROC: binary uses positive-class prob, multi-class uses one-vs-rest
        try:
            y_proba = model.predict_proba(X_val)
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
        except (AttributeError, ValueError):
            auc = float('nan')

        results.append({
            'fold': fold_idx,
            'method': 'traditional',
            'model': model_name,
            'accuracy': acc,
            'f1_score': f1,
            'auc_roc': auc,
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'seed': seed
        })

    return results


# Run all traditional ML methods as baselines
def run_all_traditional_methods(X, y, n_folds=5, seed=42):
    """Run all traditional methods"""
    methods = ['svm', 'random_forest', 'knn', 'decision_tree']
    
    if XGB_AVAILABLE:
        methods.append('xgboost')

    all_results = []

    print("="*60)
    print("Running Traditional Methods (5-Fold CV)")
    print("="*60)

    for method in methods:
        print(f"\n{method.upper()}")
        results = run_traditional_experiment(X, y, method, n_folds, seed)
        all_results.extend(results)

        accuracies = [r['accuracy'] for r in results]
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")

    print("\n" + "="*60)
    print("All traditional methods completed!")
    print("="*60)

    return all_results


# Generate summary table for traditional method results
def get_traditional_summary(results_df):
    """
    Generate summary table for traditional methods.
    
    Args:
        results_df: DataFrame from traditional experiments 
    
    Returns:
        DataFrame: Summary table 
    """
    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'precision': ['mean', 'std']
    }).round(4)

    summary.columns = [
        'acc_mean', 'acc_std', 
        'f1_mean', 'f1_std',
        'recall_mean', 'recall_std', 
        'precision_mean', 'precision_std'
    ]

    return summary
