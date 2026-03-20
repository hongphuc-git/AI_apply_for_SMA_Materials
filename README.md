# SMA AI Training CLI Bundle

Portable CLI bundle for SMA parameter regression and tabular-model experimentation.

This project lets you train multiple built-in models from the command line by passing:

- a data folder with `--data-dir`
- a result folder with `--runs-root`

## What is included

- Main CLI: `run_training.py`
- Training framework: `sma_colab_framework.py`
- Visualization helper: `visualize_trained_ann_results.py`

Built-in models:

- `ann`
- `mlp_tabular`
- `cnn`
- `resdnn`
- `tabular_resnet`
- `resdnn_v2`
- `resdnn_v3`
- `transformer`
- `xgboost`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`
- `ada_boost`

## Data path

You only need to provide your data folder path with `--data-dir`.

Example:

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results>
```

You can also override common parameters directly from the CLI without using a config file, for example `--epochs`, `--learning-rate`, `--batch-size`, `--weight-decay`, `--dropout`, and `--c-loss-weight`.

For neural models, you can also choose the training optimizer directly with `--train-optimizer` such as `adamw`, `adam`, `nadam`, `rmsprop`, or `sgd`.

## Local setup

### Download the code

Clone the repository:

```bash
git clone https://github.com/hongphuc-git/AI_apply_for_SMA_Materials.git
cd AI_apply_for_SMA_Materials
```

Or download it manually:

1. Open `https://github.com/hongphuc-git/AI_apply_for_SMA_Materials`
2. Click `Code` -> `Download ZIP`
3. Extract the ZIP file
4. Open a terminal inside the extracted `AI_apply_for_SMA_Materials` folder

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### List available models

```bash
python run_training.py --list-models
```

When training starts, the code now prints a short runtime summary including dataset split sizes, feature/target format, model architecture or estimator type, device usage, a rough training-time estimate, and a note about whether the current parameters look suitable for the detected hardware.

## Google Colab setup

Recommended Colab flow:

1. Download the repository as a ZIP from GitHub
2. Upload the ZIP file to Colab
3. Unzip it in `/content`
4. Install dependencies
5. Run training

### 1. Download ZIP from GitHub

1. Open `https://github.com/hongphuc-git/AI_apply_for_SMA_Materials`
2. Click `Code` -> `Download ZIP`
3. Save the file as `AI_apply_for_SMA_Materials.zip`

### 2. Upload ZIP to Colab

```python
from google.colab import files
uploaded = files.upload()
```

### 3. Unzip and install

```python
!unzip -q /content/AI_apply_for_SMA_Materials.zip -d /content/
%cd /content/AI_apply_for_SMA_Materials
!python -m pip install -r requirements.txt
```

### 4. Optional: mount Google Drive for dataset/results

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Train in Colab

If your dataset is in Google Drive:

```python
!python run_training.py --model resdnn_v3 --data-dir /content/drive/MyDrive/SMA_data --runs-root /content/drive/MyDrive/SMA_results
```

If your dataset is also uploaded directly into Colab:

```python
!python run_training.py --model resdnn_v3 --data-dir /content/SMA_data --runs-root /content/SMA_results
```

### Optional clone-style Colab workflow

If the repository is publicly accessible from Colab, you can also use a shorter setup like this:

```python
import os

!git clone https://github.com/hongphuc-git/AI_apply_for_SMA_Materials.git
os.chdir("AI_apply_for_SMA_Materials")
!python -m pip install -r requirements.txt
```

If `git clone` fails, use the ZIP workflow above instead.

## Quick training examples

### Default stable model

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag default-stable
```

### Tabular MLP baseline

```bash
python run_training.py --model mlp_tabular --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag mlp-tabular
```

### Tabular residual model

```bash
python run_training.py --model tabular_resnet --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag tabular-resnet
```

### Tree-based baselines

```bash
python run_training.py --model xgboost --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag xgb-baseline
python run_training.py --model hist_gradient_boosting --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag hgb-baseline
```

## Hyperparameter overrides

### Inline JSON

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --config-json '{"epochs": 300, "learning_rate": 0.00012, "batch_size": 128, "c_loss_weight": 2.8}'
```

### Direct CLI overrides

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --epochs 500 --learning-rate 0.00008 --batch-size 64
```

### Direct optimizer selection

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --epochs 500 --learning-rate 0.00008 --train-optimizer nadam
```

For optimizers that use momentum:

```bash
python run_training.py --model mlp_tabular --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --train-optimizer sgd --optimizer-momentum 0.9 --learning-rate 0.0005
```

### JSON file

Create `config.json`:

```json
{
  "epochs": 300,
  "learning_rate": 0.00012,
  "batch_size": 128,
  "c_loss_weight": 2.8
}
```

Then run:

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --config-file config.json
```

## Sweep / Bayesian optimization

### XGBoost search

```bash
python run_training.py --model xgboost --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --optimizer optuna-tpe --n-trials 12 --tag bo-xgb
```

### ResDNN v3 search

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --optimizer optuna-tpe --n-trials 10 --tag bo-resdnn-v3 --config-json '{"epochs": 220}'
```

## Output structure

Every run is saved inside the folder passed to `--runs-root`.

Example:

```bash
python run_training.py --model resdnn_v3 --data-dir D:/SMA/data --runs-root D:/SMA/results --tag default-stable
```

Example output:

```text
D:/SMA/results/
  resdnn_v3_20260320_101500_default-stable/
    run_manifest.json
    metrics.json
    predictions.npz
    loss_curve.png
    test_scatter_physical.png
    test_scatter_normalized.png
```

## Summaries and visualization

Create a leaderboard from finished runs:

```bash
python run_training.py --summarize-runs-dir <path-to-save-results>
```

This writes:

- `<path-to-save-results>/leaderboard.csv`
- `<path-to-save-results>/leaderboard.json`

Visualize a finished run:

```bash
python visualize_trained_ann_results.py --output-dir <path-to-save-results>/resdnn_v3_YYYYMMDD_HHMMSS_tag
```

## Suggested models

- `resdnn_v3`: best overall default for stable SMA regression
- `tabular_resnet`: strong built-in tabular residual model
- `mlp_tabular`: simple built-in neural baseline for tabular inputs
- `xgboost`: strong non-neural baseline
- `hist_gradient_boosting`: fast boosting baseline for larger structured datasets
- `random_forest` / `extra_trees`: quick comparison baselines
- `cnn` / `transformer`: useful when the input is treated more like sequence or structured signal data

## Built-in tabular models

Neural tabular-style options already implemented:

- `ann`
- `mlp_tabular`
- `resdnn`
- `tabular_resnet`
- `resdnn_v2`
- `resdnn_v3`

Tree and ensemble baselines already implemented:

- `xgboost`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`
- `ada_boost`

## Possible extensions

This repository is currently focused on supervised SMA regression, but the same workflow can be reused for broader ML experiments.

External models that may be worth adding later:

- `LightGBM`
- `CatBoost`
- `TabNet`
- `FT-Transformer`

### Flow matching note

This bundle is not a ready-made flow-matching implementation, but it can still serve as a workflow reference.

- Reuse the dataset path handling
- Reuse the output folder and run manifest flow
- Reuse experiment tags and result tracking
- Swap the current regression target for flow-matching supervision such as velocity, drift, or trajectory terms

## Notes

- GPU is used automatically for PyTorch models when CUDA is available.
- `xgboost` currently uses CPU-oriented defaults.
- Dataset files are not included.
- For reproducible runs, prefer passing both `--data-dir` and `--runs-root` explicitly.
