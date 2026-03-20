# SMA AI Training CLI Bundle

Portable training bundle for SMA parameter regression.

Clone or download this repository, install requirements, then run the CLI by passing your input data folder and the folder where you want to save results.

## What this bundle contains

- Multi-model CLI runner: `run_training.py`
- Advanced framework: `sma_colab_framework.py`
- Models: `ann`, `mlp_tabular`, `cnn`, `resdnn`, `tabular_resnet`, `resdnn_v2`, `resdnn_v3`, `transformer`, `xgboost`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `ada_boost`
- Visualization helper: `visualize_trained_ann_results.py`

## Data path

You only need to pass the path to your data folder with `--data-dir`.

In the default bundle workflow, this path usually points to the folder used by the training scripts for loading SMA dataset files.

## Quick start

> [!TIP]
> In the commands below, replace `<path-to-data-folder>` with your data folder path, and replace `<path-to-save-results>` with the folder where you want training outputs to be stored.

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

### 3. Install on Google Colab

Use this if you want to train directly in Colab.

1. Open a new Google Colab notebook
2. Set `Runtime` -> `Change runtime type` -> `GPU` if available
3. Download this repository as a ZIP file from GitHub first
4. Upload the ZIP file to Colab
5. Run the setup cells below

Download the code first:

1. Open `https://github.com/hongphuc-git/AI_apply_for_SMA_Materials`
2. Click `Code` -> `Download ZIP`
3. Keep the downloaded file, for example `AI_apply_for_SMA_Materials.zip`

Upload the ZIP file to Colab:

```python
from google.colab import files
uploaded = files.upload()
```

Extract the ZIP file, enter the folder, and install dependencies:

```python
!unzip -q /content/AI_apply_for_SMA_Materials.zip -d /content/
%cd /content/AI_apply_for_SMA_Materials
!python -m pip install -r requirements.txt
```

If your dataset is stored on Google Drive, mount Drive before training:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Example training command in Colab:

```python
!python run_training.py --model resdnn_v3 --data-dir /content/drive/MyDrive/SMA_data --runs-root /content/drive/MyDrive/SMA_results
```

If you also uploaded the dataset directly to Colab instead of Google Drive, you can run:

```python
!python run_training.py --model resdnn_v3 --data-dir /content/SMA_data --runs-root /content/SMA_results
```

### 4. List available models

```bash
python run_training.py --list-models
```

### 5. Train with explicit input and output paths

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
- `tabular_resnet`: good built-in tabular residual model for structured SMA features
- `mlp_tabular`: wider MLP baseline for tabular-style inputs
- `resdnn_v2`: simpler stable residual model
- `resdnn`: lighter residual baseline if you want a smaller starting point
- `cnn`: useful when your inputs behave like structured grids or image-like maps
- `ann`: simple fully connected baseline for compact tabular-style features
- `xgboost`: strong non-neural baseline
- `hist_gradient_boosting`: fast boosted-tree option for larger tabular datasets
- `gradient_boosting` / `ada_boost`: additional classical tabular boosting baselines
- `transformer`: sequence-style alternative
- `random_forest` / `extra_trees`: quick baselines

## Possible extensions

This bundle is currently documented for supervised SMA regression, but the same data-to-output workflow can also be used as a reference when adapting the project to other ML settings.

### Models you can also explore

- Built in now: `ann`, `mlp_tabular`, `resdnn`, `tabular_resnet`, `resdnn_v2`, `resdnn_v3`, `xgboost`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `ada_boost`
- `cnn` / `transformer`: useful if your SMA inputs are treated as ordered sequences, spatial maps, or time-frequency representations
- `ann` / `mlp_tabular`: suitable for compact engineered features or reduced latent vectors
- `resdnn`, `tabular_resnet`, `resdnn_v2`, `resdnn_v3`: good candidates when you want deeper residual backbones for harder nonlinear mappings
- `xgboost`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `ada_boost`: practical tabular baselines for comparing neural and non-neural performance

For tabular data specifically, many current workflows also consider external models such as:

- `LightGBM`: strong gradient-boosted tree baseline for structured tables
- `CatBoost`: often effective when categorical features or mixed tabular signals are involved
- `TabNet`: deep tabular model with feature selection-style attention
- `FT-Transformer`: transformer-style architecture designed for tabular inputs

These models are not implemented in this bundle by default, but they are reasonable next steps if you want to expand beyond the current built-in models.

### Applying the workflow to flow matching

If you want to extend the project toward flow matching, this repository can be used mainly as a workflow reference, not as a ready-made flow-matching implementation.

- Reuse the same ideas for dataset paths, training outputs, experiment tags, and saved result folders
- Replace the current regression target with the quantity needed by flow matching, such as velocity, drift, or trajectory-related supervision
- Start from `ann`, `cnn`, `resdnn`, or `transformer` as backbone candidates depending on whether your inputs are tabular, spatial, or sequential
- Keep `xgboost` or tree models as simple non-generative baselines for comparison tasks, but they are not typical backbones for flow matching itself

## Notes

- GPU is used automatically for PyTorch models when CUDA is available.
- `xgboost` currently uses CPU-oriented settings by default.
- Dataset files are not included; pass your own data folder path with `--data-dir`.
- If you omit `--data-dir`, the CLI prompts for it, but using `--data-dir` and `--runs-root` explicitly is recommended for reproducible runs.
