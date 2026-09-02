# Image-based Transformation for Small Tabular Datasets

This repository contains the code and fold-level results for a comparative study of tabular-to-image transfer learning on six datasets with fewer than 200 samples. Five image transformations and four ImageNet-pretrained CNNs are compared with five traditional classifiers under shared stratified five-fold splits.

## 1. Methods

- Transformations: TINTO, IGTD, SuperTML, DeepInsight and REFINED
- CNNs: EfficientNet V2 M, ResNeXt-50, MobileNet V3 Large and DenseNet-161
- Baselines: SVM, Random Forest, KNN, Decision Tree and XGBoost
- Metrics: accuracy, weighted F1-score and one-vs-rest AUC-ROC
- Tests: Friedman, Nemenyi post-hoc and Wilcoxon signed-rank

## 2. Datasets

The feature counts refer to the encoded CSV files used by the experiments.

| Dataset | Type | Samples | Encoded features | Classes | Class distribution |
| --- | --- | ---: | ---: | ---: | --- |
| Iris | Numerical | 150 | 4 | 3 | 50 / 50 / 50 |
| Parkinsons | Numerical | 195 | 22 | 2 | 48 / 147 |
| Hepatitis | Mixed | 155 | 19 | 2 | 32 / 123 |
| Acute Inflammations | Mixed | 120 | 11 | 2 | 61 / 59 |
| Zoo | Categorical | 101 | 21 | 7 | 41 / 20 / 5 / 13 / 4 / 8 / 10 |
| Hayes-Roth | Categorical | 160 | 15 | 3 | 65 / 64 / 31 |

The distributions were checked with `data/check_balance.py`. Each dataset directory retains its original UCI files together with the processed CSV and `dataset_info.json`. The preprocessing code is in `notebook/preprocess_dataset.ipynb`.

## 3. Repository Structure

```text
core/                         Shared loading, model, training and analysis modules
experiments/                  Experiment runners
data/                         Original data, processed CSVs and metadata
notebook/
`-- preprocess_dataset.ipynb  Dataset preprocessing
TINTOlib/                     Bundled tabular-to-image library
results/                      Included fold-level results and analysis outputs
output/images/                Included generated-image cache
```

Main entry points:

- `experiments/run_transfer.py`: image-based experiments
- `experiments/run_traditional.py`: traditional baselines
- `experiments/run_transfer_simplenorm.py`: normalization ablation
- `experiments/run_history.py`: representative training histories
- `experiments/run_analysis.py`: summaries and statistical tests

## 4. Installation

The reported environment used Python 3.11, PyTorch 2.2.2 and TorchVision 0.17.2.

```powershell
conda create -n pytorch-tabular-image python=3.11
conda activate pytorch-tabular-image
cd path\to\PyTorch_26Project
pip install -r requirements.txt
```

Check the installation and selected device:

```powershell
python -c "import torch, torchvision, sklearn, xgboost; print(torch.__version__, torchvision.__version__)"
python -c "from core.config import Config; print(Config.PROJECT_ROOT); print(Config.get_device())"
```

The local `TINTOlib/` directory is the transformation implementation used by the study. Do not install a different TINTOlib release over it.

### 4.1 REFINED Requirement

REFINED calls `mpiHill_UF.py` through `mpiexec`. On Windows, install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) and confirm that `mpiexec` is available on `PATH`. On Linux, install an MPI implementation such as MPICH. MPI is not required for the other four transformations.

## 5. Reproduction

Run commands from the repository root. The processed data, generated images and complete result CSVs are included, so the full experiment does not need to be rerun to inspect or verify the reported outputs.

### 5.1 Optional: Regenerate Processed Data

Open `notebook/preprocess_dataset.ipynb` with its working directory set to `notebook/`, then run all cells. This overwrites the processed CSV and metadata files.

### 5.2 Traditional Baselines

Set `DATASET` near the top of `main()` in `experiments/run_traditional.py`, then run:

```powershell
python experiments/run_traditional.py
```

Each dataset produces 25 fold-level results: five classifiers over five folds. Completed classifiers in the existing dataset CSV are skipped.

### 5.3 Image-based Experiments

Set `DATASET` near the top of `main()` in `experiments/run_transfer.py`, then run:

```powershell
python experiments/run_transfer.py
```

Each dataset produces 100 fold-level results:

```text
5 transformations x 4 CNNs x 5 folds = 100 runs
```

Results are appended to `results/{dataset}_transfer.csv`. Existing `(fold, method, model)` rows are skipped, including rows containing `NaN`. Remove a row before rerunning that combination.

Generated images are stored under `output/images/{dataset}/{method}/fold_N/` and reused across backbones. The checked-in cache allows the training pipeline to reuse the reported image representations. A complete dataset took approximately 1 hour and 20 minutes on the RTX 4060 Laptop GPU used for the study; runtime depends on hardware and cache state.

### 5.4 Normalization Ablation

Set `DATASET` in `experiments/run_transfer_simplenorm.py`, then run:

```powershell
python experiments/run_transfer_simplenorm.py
```

Outputs are saved as `results/{dataset}_transfer_simplenorm.csv`. The script compares these results with the corresponding main transfer CSV when both files exist.

### 5.5 Training Histories

`experiments/run_history.py` reruns the combinations listed in `RUNS` and saves their epoch histories and plots under `results/history/`.

```powershell
python experiments/run_history.py
```

### 5.6 Analysis

After the required experiment CSVs are present, run:

```powershell
python experiments/run_analysis.py
```

This creates:

- `results/analysis_summary.csv`
- `results/analysis_friedman.csv`
- `results/analysis_wilcoxon.csv`

## 6. Experimental Configuration

- Stratified five-fold cross-validation with split seed `42`
- 50 epochs, batch size `16` and learning rate `1e-4`
- Adam optimizer and `ReduceLROnPlateau` scheduler (`factor=0.5`, `patience=5`)
- Frozen ImageNet-pretrained backbone; replacement classifier head trained
- ImageNet normalization after resizing
- IGTD grid scale set to `ceil(sqrt(n_features))`
- Incremental CSV logging and generated-image caching

The image cache does not encode every preprocessing parameter. Clear the relevant cached directory before rerunning an experiment with changed preprocessing or transformation settings.

## 7. Results

The complete fold-level results are included under `results/`. The best mean accuracies are summarized below.

| Dataset | Best image-based method | Accuracy | Best baseline | Accuracy |
| --- | --- | ---: | --- | ---: |
| Iris | SuperTML + ResNeXt-50 | 0.8866 | KNN | 0.9733 |
| Parkinsons | IGTD + DenseNet-161 | 0.9180 | XGBoost | 0.9282 |
| Hepatitis | DeepInsight + MobileNet V3 Large | 0.8194 | KNN | 0.8452 |
| Acute Inflammations | DeepInsight + DenseNet-161 | 1.0000 | KNN | 1.0000 |
| Zoo | REFINED + ResNeXt-50 | 0.7919 | Random Forest | 0.9800 |
| Hayes-Roth | SuperTML + DenseNet-161 | 0.7188 | SVM | 0.8313 |

## 8. Third-party Software and Data

The project-specific pipeline is implemented in `core/` and `experiments/`. `TINTOlib/` is bundled under its original Apache License 2.0. PyTorch and TorchVision provide the pretrained CNNs; scikit-learn and XGBoost provide the baselines and evaluation utilities. The datasets retain their original UCI documentation and terms. Complete dependency versions are recorded in `requirements.txt`.

## 9. References

- B. Sun, L. Yang, W. Zhang, M. Lin, P. Dong, C. Young, and J. Dong, “SuperTML: Two-dimensional word embedding for the precognition on structured tabular data,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops*, 2019, pp. 2973–2981. [doi:10.1109/CVPRW.2019.00360](https://doi.org/10.1109/CVPRW.2019.00360)
- A. Sharma, E. Vans, D. Shigemizu, K. A. Boroevich, and T. Tsunoda, “DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture,” *Scientific Reports*, vol. 9, Art. no. 11399, 2019. [doi:10.1038/s41598-019-47765-6](https://doi.org/10.1038/s41598-019-47765-6)
- O. Bazgir, R. Zhang, S. R. Dhruba, R. Rahman, S. Ghosh, and R. Pal, “Representation of features as images with neighborhood dependencies for compatibility with convolutional neural networks,” *Nature Communications*, vol. 11, Art. no. 4391, 2020. [doi:10.1038/s41467-020-18197-y](https://doi.org/10.1038/s41467-020-18197-y)
- Y. Zhu et al., “Converting tabular data into images for deep learning with convolutional neural networks,” *Scientific Reports*, vol. 11, Art. no. 11325, 2021. [doi:10.1038/s41598-021-90923-y](https://doi.org/10.1038/s41598-021-90923-y)
- M. Castillo-Cara, R. Talla-Chumpitaz, R. García-Castro, and L. Orozco-Barbosa, “TINTO: Converting tidy data into image for classification with 2-dimensional convolutional neural networks,” *SoftwareX*, vol. 22, Art. no. 101391, 2023. [doi:10.1016/j.softx.2023.101391](https://doi.org/10.1016/j.softx.2023.101391)
- J. Liu, D. González-Fernández, M. Castillo-Cara, and R. García-Castro, “TINTOlib: A Python library for transforming tabular data into synthetic images for deep neural networks,” *SoftwareX*, vol. 32, Art. no. 102444, 2025. [doi:10.1016/j.softx.2025.102444](https://doi.org/10.1016/j.softx.2025.102444)
- M. Kelly, R. Longjohn, and K. Nottingham, “The UCI Machine Learning Repository,” 2023. [Online]. Available: [https://archive.ics.uci.edu/](https://archive.ics.uci.edu/)
- PyTorch, “Models and pre-trained weights,” *TorchVision documentation*. [Online]. Available: [https://docs.pytorch.org/vision/stable/models.html](https://docs.pytorch.org/vision/stable/models.html)
