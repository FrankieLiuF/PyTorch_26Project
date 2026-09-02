# Image-based Transformation for Small Tabular Datasets

This project investigates whether small tabular classification datasets can benefit from image-based transfer learning. Tabular samples are converted into synthetic images with TINTOlib and classified using ImageNet-pretrained convolutional neural networks (CNNs). Their performance is compared with conventional machine-learning models under the same five-fold cross-validation protocol.
 
## 1. Research Question

Can tabular-to-image transformations combined with pretrained CNNs provide competitive classification performance on small numerical, mixed, and categorical datasets when compared with traditional tabular classifiers?

The project evaluates:

- five tabular-to-image methods: TINTO, IGTD, SuperTML, REFINED, and DeepInsight;
- four pretrained CNN architectures: EfficientNet V2 M, ResNeXt-50, MobileNet V3 Large, and DenseNet-161;
- five traditional baselines: SVM, Random Forest, KNN, Decision Tree, and XGBoost;
- Accuracy, weighted F1-score, and one-vs-rest AUC-ROC;
- statistical differences using Friedman, Nemenyi post-hoc, and Wilcoxon signed-rank tests.

## 2. Experimental Pipeline

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

## 3. Datasets

Six small UCI classification datasets are included. The feature counts below refer to the preprocessed data used by the experiments.

| Dataset | Data type | Samples | Encoded features | Classes | Class distribution |
| --- | --- | ---: | ---: | ---: | --- |
| Iris | Numerical | 150 | 4 | 3 | 50 / 50 / 50 |
| Parkinsons | Numerical | 195 | 22 | 2 | 48 / 147 |
| Hepatitis | Mixed | 155 | 19 | 2 | DIE: 32; LIVE: 123 |
| Acute Inflammations | Mixed | 120 | 11 | 2 | 61 / 59 |
| Zoo | Categorical | 101 | 21 | 7 | 41 / 20 / 5 / 13 / 4 / 8 / 10 |
| Hayes-Roth | Categorical | 160 | 15 | 3 | 65 / 64 / 31 |

Feature counts refer to the encoded data used by the experiments. Except where class names are shown explicitly, class distributions follow the target-label order stored in each dataset's `dataset_info.json`. The distributions were verified using `data/check_balance.py`; its saved output is provided in `data/dataset_balance.csv`.

Encoding is dataset-specific. The 13 binary clinical indicators in Hepatitis are stored as single 0/1 columns. In Acute Inflammations, each of the five binary symptom indicators is represented by two complementary dummy columns, giving 11 encoded features from the six original attributes. Zoo retains its 15 binary attributes and one-hot encodes `legs`, while Hayes-Roth one-hot encodes all four categorical attributes. The same processed CSV is supplied to the image-based and traditional pipelines within each dataset. The complete preparation code is in `notebook/preprocess_dataset.ipynb`.

Each dataset directory retains the files supplied with the original UCI download for provenance. The preprocessing notebook reads `iris.data`, `parkinsons.data`, `hepatitis.data`, `diagnosis.data`, `zoo.data`, and both `hayes-roth.data` and `hayes-roth.test`. Other files distributed with those datasets are retained but are not used by the experimental pipeline.

## 4. Repository Structure

```text
core/                         Reusable data, model, training, baseline, and analysis modules
experiments/                  Batch runners for transfer learning, baselines, and analysis
data/                         Original UCI files, processed CSVs, and dataset metadata
notebook/
`-- preprocess_dataset.ipynb  Dataset preparation used by the experiments
TINTOlib/                     Local project copy of the tabular-to-image library
results/                      Fold-level results, summaries, and statistical tests
output/images/                Generated image cache (created automatically and gitignored)
```

The main experiment entry points are:

- `experiments/run_transfer.py`: main tabular-to-image transfer-learning experiment;
- `experiments/run_traditional.py`: traditional machine-learning baselines;
- `experiments/run_analysis.py`: summaries and statistical tests;
- `experiments/run_transfer_simplenorm.py`: additional normalization ablation;
- `experiments/run_history.py`: training-history experiment support.

## 5. Environment Setup

The project was developed with Python 3.11, PyTorch 2.2.2, and torchvision 0.17.2. A CUDA-capable GPU is recommended for the CNN experiments, although PyTorch can fall back to CPU at a substantially slower speed.

```powershell
conda create -n pytorch-tabular-image python=3.11
conda activate pytorch-tabular-image
cd path\to\PyTorch_26Project
pip install -r requirements.txt
```

Confirm that the project imports correctly before starting a long experiment:

```powershell
python -c "import torch, torchvision, sklearn, xgboost; print(torch.__version__, torchvision.__version__)"
python -c "from core.config import Config; print(Config.PROJECT_ROOT); print(Config.get_device())"
```

The processed CSV files and their `dataset_info.json` metadata are already included under `data/`, so preprocessing is not required for a normal reproduction run.

`TINTOlib/` is imported directly from the repository. Keep this bundled copy in place so that the experiments use the same implementation as the reported study. The dataset-specific IGTD grid size is supplied by the project code rather than by installing a different TINTOlib release.

### 5.1 MPI Requirement for REFINED

REFINED launches `mpiHill_UF.py` through `mpiexec`.

- Windows: install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) and confirm that `mpiexec` is available on `PATH`.
- Linux: install an MPI implementation such as MPICH (`sudo apt install mpich`).

MPI is not required when REFINED is excluded. TINTO, IGTD, SuperTML, and DeepInsight run without it.

### 5.2 Known Windows Warnings

- Conda may report that a system-wide MS-MPI installation shadows its own package. This does not prevent REFINED from running when `mpiexec` is correctly configured.
- Joblib/loky may be unable to detect the number of physical CPU cores through WMIC on Windows 11. The experiment scripts set `LOKY_MAX_CPU_COUNT` to suppress this harmless warning.

## 6. Reproducing the Experiments

Run all commands from the repository root.

### 6.1 Preprocess the Datasets

This step is only necessary when the processed CSV files are missing or the preprocessing procedure has changed.

Open `notebook/preprocess_dataset.ipynb` and run all cells. It generates a dataset CSV and `dataset_info.json` for each dataset.

Run the notebook with its working directory set to `notebook/`. It derives the repository root from the notebook's current directory. Reprocessing overwrites the existing processed CSV and metadata files, so keep the checked-in versions if the aim is to reproduce the reported results exactly.

### 6.2 Run the Traditional Baselines

Set `DATASET` near the top of `main()` in `experiments/run_traditional.py`, then run:

```powershell
python experiments/run_traditional.py
```

Each dataset evaluates five models over five folds. A model already present in the dataset's results CSV is skipped when the script is run again. To rerun an incomplete or failed model, remove all of that model's rows from the corresponding CSV first.

### 6.3 Run the Transfer-learning Experiments

Set `DATASET` near the top of `main()` in `experiments/run_transfer.py`, then run:

```powershell
python experiments/run_transfer.py
```

A complete dataset experiment contains:

```text
5 transformations x 4 CNNs x 5 folds = 100 training runs
```

Results are saved incrementally to `results/{dataset}_transfer.csv`. A restarted experiment skips previously recorded `(fold, method, model)` combinations, including rows containing `NaN`. To rerun a failed combination, remove its row from the CSV before restarting the script.

On the NVIDIA RTX 4060 Laptop GPU used for the study, a complete 100-run dataset experiment took approximately 1 hour and 20 minutes. Runtime varies with the transformation method, hardware and whether generated images are already cached. CPU execution is supported but is substantially slower.

Although `bargraph` is available in the general TINTOlib method registry, it is not part of the five-method experiment reported here.

### 6.4 Run the Normalization Ablation

The supplementary experiment compares ImageNet normalization with a simple mean and standard deviation of 0.5 for every channel. Set `DATASET` in `experiments/run_transfer_simplenorm.py`, then run:

```powershell
python experiments/run_transfer_simplenorm.py
```

Results are written to `results/{dataset}_transfer_simplenorm.csv`. If the corresponding main transfer CSV exists, the script also prints matched win, tie and loss counts and the mean accuracy difference.

### 6.5 Record Representative Training Histories

`experiments/run_history.py` reruns a small, explicitly selected set of transformation-backbone combinations and retains their epoch histories. Edit the `RUNS` list only if different representative runs are required, then run:

```powershell
python experiments/run_history.py
```

The raw histories and plotted loss curves are saved under `results/history/`. These runs are separate diagnostics and do not replace the fold-level results used by the main analysis.

### 6.6 Produce Summaries and Statistical Tests

After all required dataset experiments have completed, run:

```powershell
python experiments/run_analysis.py
```

This produces:

- `results/analysis_summary.csv`: mean and standard deviation for each method and dataset;
- `results/analysis_friedman.csv`: Friedman tests, Nemenyi comparisons, average ranks, and letter groups;
- `results/analysis_wilcoxon.csv`: paired comparison between the best transfer method and best traditional baseline.

### 6.7 Resetting Cached Outputs

Only clear outputs when preprocessing or experiment settings have changed. The image-cache path records the dataset, transformation method and fold, but not every preprocessing or transformation parameter. Images generated under old settings may therefore be reused unless the cache is cleared. These commands permanently remove generated files, so retain a copy of results that must be preserved.

```powershell
# Clear generated images only
if (Test-Path output/images) { Remove-Item -Recurse -Force output/images }

# Clear one dataset's transfer results only
Remove-Item results/iris_transfer.csv
```

Replace `iris` with the required dataset name. Removing a results CSV does not remove its generated-image cache.

## 7. Current Results

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

## 8. Evaluation Notes and Limitations

- Accuracy measures overall correctness; weighted F1 gives class-frequency-weighted precision/recall balance; AUC-ROC is calculated using the one-vs-rest convention for multiclass datasets.
- Zoo contains a class with only four samples. One validation fold therefore lacks at least one class, making multiclass AUC-ROC undefined for that fold. Its mean AUC is calculated over the remaining valid folds.
- These datasets contain only 101-195 samples. Five-fold estimates and hypothesis tests therefore have limited statistical power.
- Converting tabular data to images and training CNNs introduces substantially more computation than the traditional baselines.
- Only one fixed split seed (`42`) and the configured model settings are represented; repeated cross-validation or additional training seeds would support stronger general conclusions.
- The experiments evaluate pretrained CNNs on synthetic images, not CNNs trained from scratch, so the conclusions apply specifically to this transfer-learning setup.

## 9. Key Implementation Details

- Training uses 50 epochs, batch size 16, learning rate `1e-4`, weight decay `1e-5`, and five stratified folds.
- Generated images are resized for CNN input and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- IGTD chooses its grid scale dynamically as `ceil(sqrt(n_features))`, allowing datasets with different preprocessed feature counts to fit the image grid.
- Fold results are written incrementally so that long experiments can resume after interruption.

## 10. Code and Third-Party Software

The project-specific experiment framework is implemented in `core/` and `experiments/`. This includes dataset loading, fold-wise preprocessing, image-cache handling, model construction, training, baseline evaluation, metric logging and statistical analysis.

`TINTOlib/` is a bundled third-party library used to generate the five image representations. It retains its original Apache License 2.0 and supporting documentation. PyTorch and TorchVision provide the pretrained CNNs, while scikit-learn and XGBoost provide the traditional classifiers and evaluation utilities. See `requirements.txt` for the complete runtime dependency list.

The processed datasets remain subject to their original UCI dataset terms and citations. The repository does not claim ownership of the original datasets or the bundled TINTOlib implementation.

## 11. References

- M. Castillo-Cara et al., “TINTO: Converting Tidy Data into Image for Classification with 2-Dimensional Convolutional Neural Networks,” *SoftwareX*, 22, 101391, 2023. [https://doi.org/10.1016/j.softx.2023.101391](https://doi.org/10.1016/j.softx.2023.101391)
- TINTOlib documentation and source references are included in the local `TINTOlib/README.md`.
- Dataset descriptions and original files are provided through the UCI Machine Learning Repository materials stored under `data/`.
