from pathlib import Path

# Configuration and Path setup
class Config:
    """
    Configuration class for all experiments

    Functions:
        1. create_dirs
        2. get_device
        3. summary
        4. get_data_dir
    """

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent # Go back to project root from notebook

    # Dataset name → data subdirectory mapping
    # Lets load_dataset() find the right folder without hardcoding paths
    DATASET_DIR_MAP = {
        'iris': '1.Numerical_iris',
        'parkinsons': '2.Numerical_parkinsons',
        'hepatitis': '3.Mixed_hepatitis',
        'acute_inflammations': '4.Mixed_acute_inflammations',
        'zoo': '5.Categorical_zoo',
        'lenses': '6.Categorical_lenses',
    }

    OUTPUT_DIR = PROJECT_ROOT / 'output'
    IMAGES_DIR = OUTPUT_DIR / 'images'          # generated synthetic images (all methods)
    RESULTS_DIR = PROJECT_ROOT / 'results'       # experiment result CSVs

    # Training parameters
    IMG_SIZE = 20        # TINTO image size (library default, see SoftwareX 2025)
    INPUT_SIZE = 224     # EfficientNet input size
    NUM_CLASSES = 3      # number of classes in the dataset (Iris)
    EPOCHS = 50          # Training epochs
    BATCH_SIZE = 16      # Batch size
    LEARNING_RATE = 1e-4 # Learning rate
    WEIGHT_DECAY = 1e-5  # Weight decay
    SEED = 42            # random seed
    SAVE_MODEL = False

    # Cross validation
    N_FOLDS = 5 # 5-fold cross validation
    SHUFFLE = True

    # Dataset and method
    # Option: iris, wine, parkinsons, hepatitis, acute_inflammations, zoo, lenses
    DATASET_NAME = 'iris'
    TINTO_METHOD = 'tinto'
    MODEL_NAME = 'efficientnet_v2_m'

    # Dataset type mapping
    DATASET_TYPES = {
        'iris': 'numerical',
        'parkinsons': 'numerical',
        'hepatitis': 'mixed',
        'acute_inflammations': 'mixed',
        'zoo': 'categorical',
        'lenses': 'categorical',        
    }

    # Methods for comparison
    # TINTO methods
    TINTO_METHODS_LIST = [
        'tinto',
        'igtd',
        'supertml',
        'refined',
        'deepinsight',
        'bargraph',
    ]

    # Model list
    MODELS_LISTS = [
        'efficientnet_v2_m',
        'resnext50_32x4d',
        'mobilenet_v3_large',
        'densenet161',
    ]

    # Traditional methods for baseline comparison
    TRADITIONAL_METHODS = [
        'svm',
        'random_forest',
        'xgboost',
        'knn',
    ]

    # Primary Metrics
    PRIMARY_METRICS = [
        'accuracy', # Overall correctness
        'f1_score', # Weighted F1 for imbalanced handling
        'auc_roc', # Multi-class AUC (one-vs-rest)
        'loss_curves',
    ]

    # Visualization metrics
    VISUAL_METRICS = [
        'loss_curves', # Train/val loss and accuracy curves
    ]

    # All metrics to record (primary + detailed for analysis)
    ALL_METRICS = PRIMARY_METRICS + ['recall', 'precision']

    @classmethod
    def get_data_dir(cls, dataset_name=None):
        """
        Get the data directory for a dataset.

        Args:
            dataset_name: Dataset name (e.g. 'iris'). Defaults to Config.DATASET_NAME.

        Returns:
            Path to the dataset's data folder
        """
        if dataset_name is None:
            dataset_name = cls.DATASET_NAME
        subdir = cls.DATASET_DIR_MAP.get(dataset_name)
        if subdir is None:
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available: {list(cls.DATASET_DIR_MAP.keys())}"
            )
        return cls.PROJECT_ROOT / 'data' / subdir

    @classmethod
    def create_dirs(cls):
        """Create necessary output directories (images are created on demand per dataset/method/fold)."""
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Output directories ready: {cls.OUTPUT_DIR}")
        print(f"Results directory: {cls.RESULTS_DIR}")

    @classmethod
    def get_device(cls):
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU (GPU not available)")
        return device

    @classmethod
    def summary(cls):
        print("\n" + "=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"Project root: {cls.PROJECT_ROOT}")
        print(f"Data directory: {cls.get_data_dir()}")
        print(f"Output directory: {cls.OUTPUT_DIR}")
        print(f"Dataset: {cls.DATASET_NAME}")
        print(f"Number of classes: {cls.NUM_CLASSES}")
        print(f"TINTO method: {cls.TINTO_METHOD}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Random seed: {cls.SEED}")
        print(f"Folds: {cls.N_FOLDS}")
        print(f"Save model: {cls.SAVE_MODEL}")
        print("=" * 60)
