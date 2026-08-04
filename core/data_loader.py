"""
02+03.Data loading and TINTO image generation module

Functions:
- load_dataset(): Unified data loader for preprocessed datasets
- get_tinto_method(): Get TINTO method instance
- generate_tinto_images_for_fold(): Generate images for one fold
- create_5fold_tinto_images(): Generate images for all folds
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

from TINTOlib.tinto import TINTO
from TINTOlib.supertml import SuperTML
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.barGraph import BarGraph
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.combination import Combination
from TINTOlib.deepInsight import DeepInsight

from .config import Config


# Unified data loading interface
def load_dataset(dataset_name):
    """
    Load a preprocessed dataset from CSV + JSON.

    Args:
        dataset_name: Dataset name (e.g., 'iris', 'parkinsons')

    Returns:
        X: Feature array 
        y: Label array
        feature_names: List of feature names 
        target_names: List of class names 
    """
    # locate file — use dynamic lookup so every dataset finds its own folder
    # Set the dataset name so image cache paths become dataset-specific
    Config.DATASET_NAME = dataset_name
    folder_path = Config.get_data_dir(dataset_name)
    csv_path = folder_path / f"{dataset_name}.csv"
    info_path = folder_path / "dataset_info.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {csv_path}\n"
            f"Please run preprocess_datasets.ipynb first."
        )
 
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset info not found: {info_path}\n"
            f"Please run preprocess_datasets.ipynb first."
        )    

    # read file
    df = pd.read_csv(csv_path)
    with open(info_path, 'r') as f:
        info = json.load(f)

    # seperate feature and target
    target_col = info.get('target_column', 'target')
    cols_to_drop = ['id', target_col]
    X_cols = [col for col in df.columns if col not in cols_to_drop]

    X = df[X_cols].values.astype(np.float32)
    y = df[target_col].values

    feature_names = info.get('feature_names', X_cols)
    target_names = info.get('class_names', [str(i) for i in np.unique(y)])

    # print the information
    _print_dataset_info(df, X, target_names, feature_names)

    return X, y, feature_names, target_names


def _print_dataset_info(df, X, target_names, feature_names):
    """Print dataset information in unified format."""
    shape = df.shape
    print(f"\nData loaded successfully!")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(target_names)}")
    print(f"Classes: {target_names}")
    print(f"Features: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
    print(f"\nShape: {shape}")
    print(f"\nFirst 5 lines:")
    print(df.head())


# TINTO Method Functions
# Get TINTO method instance by name (all parameters use library defaults, see SoftwareX 2025)
def get_tinto_method(method_name='tinto'):
    """
    Get TINTO method instance by name.
    
    Args:
        method_name: Name of the TINTO method 
    
    Returns:
        TINTO method instance 
    """
    # Registry of all available TINTOlib image transformation methods
    methods = {
        'tinto': {
            'class': TINTO,
            'params': {'problem': 'classification', 
                       'pixels': 20,  # default pixels=20
                      'random_seed': Config.SEED}
        },
        'igtd': {
            'class': IGTD,
            'params': {
                'problem': 'classification', 
                'scale': [6, 6],  # default
                'fea_dist_method': 'Pearson', 
                'image_dist_method': 'Euclidean',
                'error': 'squared', 
                'max_step': 1000, 
                'val_step': 50,
                'random_seed': Config.SEED}
        },
        'supertml': {
            'class': SuperTML,
            'params': {
                'problem': 'classification', 
                'font_size': 10,  # default
                'feature_importance': False, 
                'random_seed': Config.SEED}
        },
        'refined': {
            'class': REFINED,
            'params': {'problem': 'classification', 'random_seed': Config.SEED}
        },
        'bargraph': {
            'class': BarGraph,
            'params': {'problem': 'classification', 'zoom': 1}  # default zoom=1
        },
        'deepinsight': {
            'class': DeepInsight,
            'params': {'problem': 'classification', 'image_dim': 20,
                       'random_seed': Config.SEED}
        },
        'distancematrix': {
            'class': DistanceMatrix,
            'params': {'problem': 'classification', 'zoom': 1}  # default zoom=1
        },
        'combination': {
            'class': Combination,
            'params': {'problem': 'classification', 'zoom': 1}  # default zoom=1
        }
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    info = methods[method_name]
    return info['class'](**info['params'])


# Generate TINTO images for one cross-validation fold
def generate_tinto_images_for_fold(X_train, y_train, X_val, y_val,
                                   fold_idx, method_name='tinto',
                                   images_folder=None, standardize=True,
                                   feature_names=None):
    """
    Generate TINTO images for a single fold.
    
    Args:
        X_train: Training features
        y_train: Training labels 
        X_val: Validation features 
        y_val: Validation labels 
        fold_idx: Fold index
        method_name: TINTO method name 
        images_folder: Path to save images 
        standardize: Whether to standardize data 
        feature_names: Feature names 
    
    Returns:
        dict: Training and validation image information 
    """

    if images_folder is None:
        # Isolate cache by dataset + method + fold
        # so switching datasets/methods never silently reuses old images
        images_folder = (
            Config.IMAGES_DIR / Config.DATASET_NAME / method_name /
            f'fold_{fold_idx}'
        )

    if feature_names is None:
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Standardize (same as traditional methods)
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train

    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['target'] = y_val

    # Get TINTO model
    tinto_model = get_tinto_method(method_name)

    # Generate training images
    train_img_folder = images_folder / 'train'
    train_img_folder.mkdir(parents=True, exist_ok=True)

    # Skip if images already cached (checked per dataset/method/fold)
    if not (train_img_folder / 'classification.csv').exists():
        print(f"  → Generating training images (Fold {fold_idx})...")
        tinto_model.fit_transform(train_df, str(train_img_folder))

    # Generate validation images
    val_img_folder = images_folder / 'val'
    val_img_folder.mkdir(parents=True, exist_ok=True)

    if not (val_img_folder / 'classification.csv').exists():
        print(f"  → Generating validation images (Fold {fold_idx})...")
        tinto_model.transform(val_df, str(val_img_folder))

    # Read and update image paths
    train_img_paths = pd.read_csv(train_img_folder / 'classification.csv')
    train_img_paths['images'] = train_img_paths['images'].apply(
        lambda x: str(train_img_folder / x))

    val_img_paths = pd.read_csv(val_img_folder / 'classification.csv')
    val_img_paths['images'] = val_img_paths['images'].apply(
        lambda x: str(val_img_folder / x))

    return {
        'train': {'images': train_img_paths, 'labels': y_train},
        'val': {'images': val_img_paths, 'labels': y_val}
    }


# Generate TINTO images for all 5 folds using stratified cross-validation
def create_5fold_tinto_images(X, y, method_name='tinto', standardize=True,
                              feature_names=None):
    """
    Generate 5-fold cross-validation TINTO images.
    
    Args:
        X: Feature data 
        y: Label data 
        method_name: TINTO method name 
        standardize: Whether to standardize data 
        feature_names: Feature names 
    
    Returns:
        dict: Dictionary containing all folds' image information 
    """
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, 
        shuffle=Config.SHUFFLE,
        random_state=Config.SEED
    )

    all_folds_data = {}

    print(f"\n=== Generating {Config.N_FOLDS}-fold TINTO Images ===")
    print(f"Method: {method_name}")
    print(f"Standardize data: {standardize}")
    print("-" * 50)

    # Iterate over 5 stratified folds: each sample appears in validation exactly once
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold_idx + 1}/{Config.N_FOLDS}:")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

        fold_data = generate_tinto_images_for_fold(
            X_train, y_train, X_val, y_val,
            fold_idx, method_name, standardize=standardize,
            feature_names=feature_names
        )

        all_folds_data[f'fold_{fold_idx}'] = fold_data

    print("\n" + "=" * 50)
    print(f" All {Config.N_FOLDS} folds generated!")

    return all_folds_data
