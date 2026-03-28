"""train_best_trial.py — retrain the best Optuna trial with a stronger final config.

Reads best_params from a sweep's study_summary.json, applies them on top of a
hardened training config (more epochs, lower min-LR, tighter patience), and
runs a single full training via the framework.

Usage
-----
python train_best_trial.py \\
    --sweep-dir /content/drive/MyDrive/sma_results/optuna-tpe-resdnn_v3_20260328_120000 \\
    --data-dir  /content/drive/MyDrive/sma_vector_ann_dataset_test/data \\
    --runs-root /content/drive/MyDrive/sma_results

python train_best_trial.py \\
    --sweep-dir /content/drive/MyDrive/sma_results/optuna-tpe-resdnn_v3_20260328_120000 \\
    --data-dir  /content/drive/MyDrive/sma_vector_ann_dataset_test/data \\
    --runs-root /content/drive/MyDrive/sma_results \\
    --epochs 600 --tag final-best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Overrides applied on top of best trial params for the final full run.
# Values are intentionally more conservative than the sweep defaults:
# longer schedule gives the model time to converge from a better starting
# point, lower min-LR squeezes more out of the cosine tail, and a longer
# warmup stabilises the wider LR ranges the sweep may have selected.
FINAL_CONFIG_UPGRADES: dict[str, object] = {
    "epochs": 500,
    "early_stopping_patience": 80,
    "min_learning_rate": 4e-6,
    "warmup_epochs": 20,
    "checkpoint_every_epochs": 20,
}


def load_best_params(sweep_dir: Path) -> dict[str, object]:
    summary_file = sweep_dir / "study_summary.json"
    if not summary_file.is_file():
        sys.exit(
            f"ERROR: study_summary.json not found in {sweep_dir}\n"
            "Point --sweep-dir at the sweep output folder (the one containing study_summary.json)."
        )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    best_params: dict[str, object] = summary.get("best_params", {})
    if not best_params:
        sys.exit(
            "ERROR: study_summary.json exists but best_params is empty — was the sweep completed?"
        )
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the best Optuna trial with a stronger final configuration."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Path to the sweep output folder containing study_summary.json",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to folder containing train.mat and test.mat",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Directory for timestamped run outputs (default: runs/)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Bundle root (defaults to the folder containing this script)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="best-trial-final",
        help="Tag appended to the output run name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count (overrides the FINAL_CONFIG_UPGRADES default of 500)",
    )
    parser.add_argument(
        "--train-optimizer",
        choices=("adamw", "adam", "nadam", "rmsprop", "sgd"),
        default="nadam",
        help="Optimizer algorithm for the final run (default: nadam)",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Extra JSON string of config overrides applied last (highest priority)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.expanduser().resolve()
    root = args.root.resolve()

    best_params = load_best_params(sweep_dir)
    print("Best trial params from sweep:")
    for key, value in sorted(best_params.items()):
        print(f"  {key}: {value}")

    overrides: dict[str, object] = {}
    overrides.update(best_params)
    overrides.update(FINAL_CONFIG_UPGRADES)
    overrides["optimizer_name"] = args.train_optimizer

    if args.data_dir:
        overrides["explicit_data_dir"] = args.data_dir
    else:
        entered = input(
            "Enter path to folder containing train.mat and test.mat: "
        ).strip()
        if not entered:
            sys.exit("ERROR: A data directory is required.")
        overrides["explicit_data_dir"] = entered

    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    if args.config_json:
        overrides.update(json.loads(args.config_json))

    print("\nFinal training config overrides:")
    for key, value in sorted(overrides.items()):
        print(f"  {key}: {value}")
    print()

    try:
        from sma_colab_framework import run_experiment
    except ImportError as exc:
        sys.exit(
            f"ERROR: Could not import sma_colab_framework: {exc}\n"
            "Make sure this script is run from the AI_apply_for_SMA_Materials folder."
        )

    run_dir = run_experiment(
        model_name="resdnn_v3",
        root=root,
        runs_root=args.runs_root,
        tag=args.tag,
        overrides=overrides,
    )
    print(f"\nFinal run saved to: {run_dir}")


if __name__ == "__main__":
    main()
