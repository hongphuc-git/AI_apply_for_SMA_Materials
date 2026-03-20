# SMA AI Training CLI Bundle

Portable training bundle for SMA parameter regression.

Clone or download this repository, install requirements, then run the CLI by passing your input data folder and the folder where you want to save results.

## What this bundle contains

- Multi-model CLI runner: `run_training.py`
- Advanced framework: `sma_colab_framework.py`
- Models: `ann`, `cnn`, `resdnn`, `resdnn_v2`, `resdnn_v3`, `transformer`, `xgboost`, `random_forest`, `extra_trees`
- Visualization helper: `visualize_trained_ann_results.py`

## Required dataset layout

Your data folder must contain:

```text
your_data_folder/
  train.mat
  test.mat
```

The CLI requires this path and passes it directly into the training config.

## Quick start

> [!TIP]
> In the commands below, replace `<path-to-data-folder>` with the folder containing `train.mat` and `test.mat`, and replace `<path-to-save-results>` with the folder where you want training outputs to be stored.

### 1. Download and enter the folder

```bash
git clone https://github.com/hongphuc-git/AI_apply_for_SMA_Materials.git
cd AI_apply_for_SMA_Materials
```

Or download it from GitHub:

1. Open `https://github.com/hongphuc-git/AI_apply_for_SMA_Materials`
2. Click `Code` -> `Download ZIP`
3. Extract the ZIP file
4. Open a terminal inside the extracted `AI_apply_for_SMA_Materials` folder

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. List available models

```bash
python run_training.py --list-models
```

### 4. Train with explicit input and output paths

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results>
```

Example:

```bash
python run_training.py --model resdnn_v3 --data-dir D:/SMA/data --runs-root D:/SMA/results
```

## Default recommended commands

### Stable high-accuracy default

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag default-stable
```

### Slightly lighter stable model

```bash
python run_training.py --model resdnn_v2 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag stable-v2
```

### Fast tree-based baseline

```bash
python run_training.py --model xgboost --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --tag xgb-baseline
```

## Override hyperparameters

### Inline JSON

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --config-json '{"epochs": 300, "learning_rate": 0.00012, "batch_size": 128, "c_loss_weight": 2.8}'
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

## PowerShell examples

```powershell
python run_training.py --model resdnn_v3 --data-dir D:/data/sma --runs-root D:/results/sma --config-json '{"epochs": 300, "learning_rate": 0.00012, "batch_size": 128, "c_loss_weight": 2.8}'
```

If PowerShell quoting is annoying, use `--config-file` instead.

## Sweep / Bayesian optimization

### XGBoost BO

```bash
python run_training.py --model xgboost --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --optimizer optuna-tpe --n-trials 12 --tag bo-xgb
```

### ResDNN v3 BO

```bash
python run_training.py --model resdnn_v3 --data-dir <path-to-data-folder> --runs-root <path-to-save-results> --optimizer optuna-tpe --n-trials 10 --tag bo-resdnn-v3 --config-json '{"epochs": 220}'
```

## Output structure

Each run is saved into a timestamped folder inside the directory passed to `--runs-root`.

Example command:

```bash
python run_training.py --model resdnn_v3 --data-dir D:/SMA/data --runs-root D:/SMA/results --tag default-stable
```

Example output structure:

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

## Create a leaderboard across finished runs

```bash
python run_training.py --summarize-runs-dir <path-to-save-results>
```

This writes:

- `<path-to-save-results>/leaderboard.csv`
- `<path-to-save-results>/leaderboard.json`

## Visualize a finished run

```bash
python visualize_trained_ann_results.py --output-dir <path-to-save-results>/resdnn_v3_YYYYMMDD_HHMMSS_tag
```

## Suggested models

- `resdnn_v3`: best default if you care about stable validation and target `C`
- `resdnn_v2`: simpler stable residual model
- `xgboost`: strong non-neural baseline
- `transformer`: sequence-style alternative
- `random_forest` / `extra_trees`: quick baselines

## Notes

- GPU is used automatically for PyTorch models when CUDA is available.
- `xgboost` currently uses CPU-oriented settings by default.
- The bundle expects `train.mat` and `test.mat` only; dataset files are not included.
- If you omit `--data-dir`, the CLI prompts for it, but using `--data-dir` and `--runs-root` explicitly is recommended for reproducible runs.
