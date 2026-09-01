# Image-based Transformation for Small Tabular Datasets

This project investigates whether small tabular classification datasets can benefit from image-based transfer learning. Tabular samples are converted into synthetic images with TINTOlib and classified using ImageNet-pretrained convolutional neural networks (CNNs). Their performance is compared with conventional machine-learning models under the same five-fold cross-validation protocol.
 
## Research Question

Can tabular-to-image transformations combined with pretrained CNNs provide competitive classification performance on small numerical, mixed, and categorical datasets when compared with traditional tabular classifiers?

The project evaluates:

- five tabular-to-image methods: TINTO, IGTD, SuperTML, REFINED, and DeepInsight;
- four pretrained CNN architectures: EfficientNet V2 M, ResNeXt-50, MobileNet V3 Large, and DenseNet-161;
- five traditional baselines: SVM, Random Forest, KNN, Decision Tree, and XGBoost;
- Accuracy, weighted F1-score, and one-vs-rest AUC-ROC;
- statistical differences using Friedman, Nemenyi post-hoc, and Wilcoxon signed-rank tests.

## Experimental Pipeline

```text
Raw UCI data
    -> preprocessing and encoding
    -> stratified five-fold cross-validation
    -> StandardScaler fitted on each training fold only
    -> tabular-to-image transformation
    -> ImageNet normalization
    -> pretrained CNN training and evaluation
    -> comparison with traditional ML baselines
    -> summary tables and statistical tests
```

Fitting the scaler separately within each training fold prevents information from the validation fold leaking into training. The same scaled fold data are used by the image-based and traditional pipelines.

## Datasets

Six small UCI classification datasets are included. The feature counts below refer to the preprocessed data used by the experiments.

| Dataset | Data type | Samples | Features | Classes |
| --- | --- | ---: | ---: | ---: |
| Iris | Numerical | 150 | 4 | 3 |
| Parkinsons | Numerical | 195 | 22 | 2 |
| Hepatitis | Mixed | 155 | 19 | 2 |
| Acute Inflammations | Mixed | 120 | 11 | 2 |
| Zoo | Categorical | 101 | 21 | 7 |
| Hayes-Roth | Categorical | 160 | 15 | 3 |

Encoding is dataset-specific. Hepatitis binary indicators remain single 0/1 columns, while the binary Acute Inflammations indicators, `legs` in Zoo, and the categorical attributes in Hayes-Roth are one-hot encoded. The preprocessing notebook records the complete preparation procedure.

## Repository Structure

```text
core/               Reusable data, model, training, baseline, and analysis modules
experiments/        Batch runners for transfer learning, baselines, and analysis
data/               Preprocessed datasets and dataset metadata
notebook/           Preprocessing notebook and pipeline demonstrations
scripts/            Experiment-design and report-planning notes
TINTOlib/           Local project copy of the tabular-to-image library
results/            Fold-level results, summaries, and statistical tests
output/images/      Generated image cache (created automatically and gitignored)
```

The main experiment entry points are:

- `experiments/run_transfer.py`: main tabular-to-image transfer-learning experiment;
- `experiments/run_traditional.py`: traditional machine-learning baselines;
- `experiments/run_analysis.py`: summaries and statistical tests;
- `experiments/run_transfer_simplenorm.py`: additional normalization ablation;
- `experiments/run_history.py`: training-history experiment support.

## Environment Setup

The project was developed with Python 3.11, PyTorch 2.2.2, and torchvision 0.17.2. A CUDA-capable GPU is recommended for the CNN experiments, although PyTorch can fall back to CPU at a substantially slower speed.

```powershell
conda create -n pytorch-tabular-image python=3.11
conda activate pytorch-tabular-image
pip install -r requirements.txt
```

`TINTOlib/` is imported directly from the repository and should not be replaced by a separately installed TINTOlib version because the local copy contains project-specific adjustments.

### MPI Requirement for REFINED

REFINED launches `mpiHill_UF.py` through `mpiexec`.

- Windows: install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) and confirm that `mpiexec` is available on `PATH`.
- Linux: install an MPI implementation such as MPICH (`sudo apt install mpich`).

MPI is not required when REFINED is excluded. TINTO, IGTD, SuperTML, and DeepInsight run without it.

### Known Windows Warnings

- Conda may report that a system-wide MS-MPI installation shadows its own package. This does not prevent REFINED from running when `mpiexec` is correctly configured.
- Joblib/loky may be unable to detect the number of physical CPU cores through WMIC on Windows 11. The experiment scripts set `LOKY_MAX_CPU_COUNT` to suppress this harmless warning.

## Reproducing the Experiments

Run all commands from the repository root.

### 1. Preprocess the datasets

This step is only necessary when the processed CSV files are missing or the preprocessing procedure has changed.

Open `notebook/preprocess_dataset.ipynb` and run all cells. It generates a dataset CSV and `dataset_info.json` for each dataset.

### 2. Run the traditional baselines

Set `DATASET` near the top of `main()` in `experiments/run_traditional.py`, then run:

```powershell
python experiments/run_traditional.py
```

Each dataset evaluates five models over five folds. Existing completed models are skipped when the script is run again.

### 3. Run the transfer-learning experiments

Set `DATASET` near the top of `main()` in `experiments/run_transfer.py`, then run:

```powershell
python experiments/run_transfer.py
```

A complete dataset experiment contains:

```text
5 transformations x 4 CNNs x 5 folds = 100 training runs
```

Results are saved incrementally to `results/{dataset}_transfer.csv`. A restarted experiment skips completed `(fold, method, model)` combinations.

Although `bargraph` is available in the general TINTOlib method registry, it is not part of the five-method experiment reported here.

### 4. Produce summaries and statistical tests

After all required dataset experiments have completed, run:

```powershell
python experiments/run_analysis.py
```

This produces:

- `results/analysis_summary.csv`: mean and standard deviation for each method and dataset;
- `results/analysis_friedman.csv`: Friedman tests, Nemenyi comparisons, average ranks, and letter groups;
- `results/analysis_wilcoxon.csv`: paired comparison between the best transfer method and best traditional baseline.

### Resetting Cached Outputs

Only clear outputs when preprocessing or experiment settings have changed. These commands permanently remove generated files, so retain a copy of results that must be preserved.

```powershell
# Clear generated images only
Remove-Item -Recurse -Force output/images

# Clear one dataset's transfer results only
Remove-Item results/{dataset}_transfer.csv
```

## Current Results

The checked-in result files show the following best mean accuracies across five folds:

| Dataset | Best image-based method | Accuracy | Best traditional method | Accuracy |
| --- | --- | ---: | --- | ---: |
| Iris | SuperTML + ResNeXt-50 | 0.8866 | KNN | 0.9733 |
| Parkinsons | IGTD + DenseNet-161 | 0.9180 | XGBoost | 0.9282 |
| Hepatitis | DeepInsight + MobileNet V3 Large | 0.8194 | KNN | 0.8452 |
| Acute Inflammations | DeepInsight + DenseNet-161 | 1.0000 | KNN | 1.0000 |
| Zoo | REFINED + ResNeXt-50 | 0.7919 | Random Forest | 0.9800 |
| Hayes-Roth | SuperTML + DenseNet-161 | 0.7188 | SVM | 0.8313 |

These results indicate that traditional models remain stronger on most of the small datasets. Image-based transfer learning is competitive on Parkinsons and matches the best baseline on Acute Inflammations, but it does not consistently improve performance. In the current per-dataset Wilcoxon comparisons, none of the best-transfer versus best-traditional differences reaches the `p < 0.05` threshold. This should be interpreted cautiously because each comparison contains only five paired folds.

The full fold-level results and metric-specific analyses are available in `results/`. The normalization-ablation CSVs are supplementary and are not included by `run_analysis.py` in the main comparison.

## Evaluation Notes and Limitations

- Accuracy measures overall correctness; weighted F1 gives class-frequency-weighted precision/recall balance; AUC-ROC is calculated using the one-vs-rest convention for multiclass datasets.
- Zoo contains a class with only four samples. One validation fold therefore lacks at least one class, making multiclass AUC-ROC undefined for that fold. Its mean AUC is calculated over the remaining valid folds.
- These datasets contain only 101-195 samples. Five-fold estimates and hypothesis tests therefore have limited statistical power.
- Converting tabular data to images and training CNNs introduces substantially more computation than the traditional baselines.
- Only one fixed random seed (`42`) and the configured model settings are represented; repeated cross-validation or additional seeds would support stronger general conclusions.
- The experiments evaluate pretrained CNNs on synthetic images, not CNNs trained from scratch, so the conclusions apply specifically to this transfer-learning setup.

## Key Implementation Details

- Training uses 50 epochs, batch size 16, learning rate `1e-4`, weight decay `1e-5`, and five stratified folds.
- Generated images are resized for CNN input and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- IGTD chooses its grid scale dynamically as `ceil(sqrt(n_features))`, allowing datasets with different preprocessed feature counts to fit the image grid.
- Fold results are written incrementally so that long experiments can resume after interruption.

## References

- M. Castillo-Cara et al., “TINTO: Converting Tidy Data into Image for Classification with 2-Dimensional Convolutional Neural Networks,” *SoftwareX*, 22, 101391, 2023. [https://doi.org/10.1016/j.softx.2023.101391](https://doi.org/10.1016/j.softx.2023.101391)
- TINTOlib documentation and source references are included in the local `TINTOlib/README.md`.
- Dataset descriptions and original files are provided through the UCI Machine Learning Repository materials stored under `data/`.
