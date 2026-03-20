from __future__ import annotations

import argparse
import json
from pathlib import Path

from sma_colab_framework import (
    MODEL_REGISTRY,
    load_overrides,
    load_search_space,
    optimize_experiment,
    print_models,
    run_experiment,
    write_leaderboard,
)


def resolve_data_dir(data_dir_arg: str | None) -> str:
    if data_dir_arg:
        return data_dir_arg
    entered = input("Enter path to folder containing train.mat and test.mat: ").strip()
    if not entered:
        raise ValueError("A data directory is required.")
    return entered


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable SMA AI training CLI for GitHub use.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Bundle root")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to folder containing train.mat and test.mat")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="resdnn_v3", help="Model to train")
    parser.add_argument(
        "--optimizer",
        choices=("none", "optuna-tpe", "optuna-random"),
        default="none",
        help="Single-run or hyperparameter optimization backend",
    )
    parser.add_argument("--runs-root", type=str, default="runs", help="Directory for timestamped outputs")
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag")
    parser.add_argument("--config-json", type=str, default=None, help="JSON string of config overrides")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to JSON config overrides")
    parser.add_argument("--search-space-json", type=str, default=None, help="JSON string overriding optimizer search space")
    parser.add_argument("--search-space-file", type=Path, default=None, help="Path to JSON optimizer search space")
    parser.add_argument("--n-trials", type=int, default=12, help="Number of optimization trials")
    parser.add_argument("--metric-path", type=str, default="val_metrics_norm.mean_rmse", help="Metric path used for sweep objective")
    parser.add_argument("--direction", choices=("minimize", "maximize"), default="minimize", help="Optimization direction")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--summarize-runs-dir", type=Path, default=None, help="Generate leaderboard files for an existing runs directory")
    args = parser.parse_args()

    if args.list_models:
        print_models()
        return

    if args.summarize_runs_dir is not None:
        leaderboard = write_leaderboard(args.summarize_runs_dir.resolve(), args.metric_path, args.direction)
        print(f"Saved leaderboard -> {leaderboard}")
        return

    overrides = load_overrides(args.config_json, args.config_file)
    overrides["explicit_data_dir"] = resolve_data_dir(args.data_dir)
    root = args.root.resolve()
    if args.optimizer == "none":
        run_experiment(
            model_name=args.model,
            root=root,
            runs_root=args.runs_root,
            tag=args.tag,
            overrides=overrides,
        )
        return

    search_space = load_search_space(args.model, args.search_space_json, args.search_space_file)
    optimize_experiment(
        model_name=args.model,
        root=root,
        runs_root=args.runs_root,
        optimizer_name=args.optimizer,
        tag=args.tag,
        fixed_overrides=overrides,
        search_space=search_space,
        n_trials=args.n_trials,
        metric_path=args.metric_path,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()
