from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from train_sma_ann import MatDatasetLoader, MinMaxNormalizer, RegressionMetrics, SMAAnnConfig

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


@dataclass
class SMAXGBoostConfig(SMAAnnConfig):
    output_dir_name: str = "python_xgboost_outputs"
    n_estimators: int = 1200
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_weight: float = 2.0
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: int = 80
    n_jobs: int = -1


class SMAXGBoostTrainer:
    def __init__(self, root: Path, config: SMAXGBoostConfig | None = None) -> None:
        self.root = root
        self.config = config or SMAXGBoostConfig()
        self.loader = MatDatasetLoader(root, self.config)
        self.data_dir = self.loader.find_data_dir()
        self.output_dir = self.root / self.config.output_dir_name
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir = self.output_dir / "sma_xgboost_regressor"
        self.model_dir.mkdir(exist_ok=True)
        self.input_normalizer = MinMaxNormalizer()
        self.target_normalizer = MinMaxNormalizer(
            np.asarray(self.config.target_min, dtype=np.float64),
            np.asarray(self.config.target_max, dtype=np.float64),
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.models: dict[str, Any] = {}
        self.history = {"train_loss": [], "val_loss": [], "per_target": {}}

    def run(self) -> None:
        train_raw = self.loader.load_split(self.data_dir / "train.mat")
        test_raw = self.loader.load_split(self.data_dir / "test.mat")
        split_data = self.prepare_datasets(train_raw, test_raw)
        self.train_model(split_data)
        artifacts = self.evaluate(split_data)
        self.save_artifacts(artifacts)
        self.save_plots(artifacts)
        self.print_summary(artifacts)

    def prepare_datasets(self, train_raw: dict[str, np.ndarray], test_raw: dict[str, np.ndarray]) -> dict[str, Any]:
        x_train_all = train_raw["X"].astype(np.float64)
        y_train_all = train_raw["Y"].astype(np.float64)
        rate_train_all = train_raw["rate"].astype(np.float64)
        x_test = test_raw["X"].astype(np.float64)
        y_test = test_raw["Y"].astype(np.float64)
        rate_test = test_raw["rate"].astype(np.float64)

        n_train_all = x_train_all.shape[0]
        n_val = max(1, round(self.config.validation_fraction * n_train_all))
        n_train_core = n_train_all - n_val
        train_idx, val_idx = self.stratified_split(rate_train_all, n_train_core, n_val)

        x_train = x_train_all[train_idx]
        y_train = y_train_all[train_idx]
        rate_train = rate_train_all[train_idx]
        x_val = x_train_all[val_idx]
        y_val = y_train_all[val_idx]
        rate_val = rate_train_all[val_idx]

        x_train = np.column_stack([x_train, rate_train]).astype(np.float32)
        x_val = np.column_stack([x_val, rate_val]).astype(np.float32)
        x_test = np.column_stack([x_test, rate_test]).astype(np.float32)

        x_train_norm = self.input_normalizer.fit(x_train).transform(x_train)
        x_val_norm = self.input_normalizer.transform(x_val)
        x_test_norm = self.input_normalizer.transform(x_test)
        y_train_norm = self.target_normalizer.transform(y_train).astype(np.float32)
        y_val_norm = self.target_normalizer.transform(y_val).astype(np.float32)
        y_test_norm = self.target_normalizer.transform(y_test).astype(np.float32)

        return {
            "epsTarget": train_raw["epsTarget"],
            "x_train": x_train,
            "y_train": y_train,
            "rate_train": rate_train,
            "x_val": x_val,
            "y_val": y_val,
            "rate_val": rate_val,
            "x_test": x_test,
            "y_test": y_test,
            "rate_test": rate_test,
            "x_train_norm": x_train_norm,
            "y_train_norm": y_train_norm,
            "x_val_norm": x_val_norm,
            "y_val_norm": y_val_norm,
            "x_test_norm": x_test_norm,
            "y_test_norm": y_test_norm,
        }

    def stratified_split(self, rate: np.ndarray, train_target: int, val_target: int) -> tuple[np.ndarray, np.ndarray]:
        rate_levels = np.unique(rate)
        train_counts = self.distribute_counts(train_target, rate_levels.size)
        val_counts = self.distribute_counts(val_target, rate_levels.size)
        tol = 1e-6
        train_index_list: list[int] = []
        val_index_list: list[int] = []
        for idx, level in enumerate(rate_levels):
            idx_rate = np.where(np.abs(rate - level) < tol)[0]
            idx_rate = self.rng.permutation(idx_rate)
            n_train_rate = int(train_counts[idx])
            n_val_rate = int(val_counts[idx])
            n_needed = n_train_rate + n_val_rate
            if idx_rate.size < n_needed:
                raise ValueError(f"Not enough samples for rate {level:.4f}. Need {n_needed}, found {idx_rate.size}.")
            train_index_list.extend(idx_rate[:n_train_rate].tolist())
            val_index_list.extend(idx_rate[n_train_rate:n_needed].tolist())
        train_indices = self.rng.permutation(np.asarray(train_index_list, dtype=np.int64))
        val_indices = self.rng.permutation(np.asarray(val_index_list, dtype=np.int64))
        return train_indices, val_indices

    @staticmethod
    def distribute_counts(total_count: int, n_bins: int) -> np.ndarray:
        base_count = total_count // n_bins
        counts = np.full(n_bins, base_count, dtype=np.int64)
        remainder = total_count - base_count * n_bins
        if remainder > 0:
            counts[:remainder] += 1
        return counts

    def build_model(self) -> Any:
        try:
            xgboost_module = importlib.import_module("xgboost")
        except ImportError as exc:
            raise ImportError(
                "xgboost is required to run this trainer. Install it with `pip install xgboost`."
            ) from exc
        xgb_regressor_cls = getattr(xgboost_module, "XGBRegressor")

        return xgb_regressor_cls(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            gamma=self.config.gamma,
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=self.config.seed,
            n_jobs=self.config.n_jobs,
        )

    def train_model(self, split_data: dict[str, Any]) -> None:
        for target_idx, label_name in enumerate(self.config.label_names):
            model = self.build_model()
            y_train_target = split_data["y_train_norm"][:, target_idx]
            y_val_target = split_data["y_val_norm"][:, target_idx]
            model.fit(
                split_data["x_train_norm"],
                y_train_target,
                eval_set=[
                    (split_data["x_train_norm"], y_train_target),
                    (split_data["x_val_norm"], y_val_target),
                ],
                verbose=False,
            )
            self.models[label_name] = model

            evals_result = model.evals_result()
            train_curve = [float(value) for value in evals_result.get("validation_0", {}).get("rmse", [])]
            val_curve = [float(value) for value in evals_result.get("validation_1", {}).get("rmse", [])]
            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is None:
                best_iteration = max(len(val_curve), 1) - 1
            self.history["per_target"][label_name] = {
                "train_rmse": train_curve,
                "val_rmse": val_curve,
                "best_iteration": int(best_iteration),
            }
            best_val_rmse = val_curve[int(best_iteration)] if val_curve else float("nan")
            print(
                f"Target {label_name:>3s} | best_iteration={int(best_iteration):4d} | "
                f"best_val_rmse={best_val_rmse:.6e}"
            )

        self.history["train_loss"] = self.aggregate_history("train_rmse")
        self.history["val_loss"] = self.aggregate_history("val_rmse")

    def aggregate_history(self, key: str) -> list[float]:
        curves = [
            np.asarray(target_history[key], dtype=np.float64)
            for target_history in self.history["per_target"].values()
            if target_history[key]
        ]
        if not curves:
            return []
        max_len = max(curve.size for curve in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for idx, curve in enumerate(curves):
            padded[idx, : curve.size] = curve
        return np.nanmean(padded, axis=0).tolist()

    def predict(self, x: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for label_name in self.config.label_names:
            model = self.models[label_name]
            outputs.append(np.asarray(model.predict(x), dtype=np.float32).reshape(-1, 1))
        return np.concatenate(outputs, axis=1)

    def evaluate(self, split_data: dict[str, Any]) -> dict[str, Any]:
        y_train_pred_norm = self.predict(split_data["x_train_norm"])
        y_val_pred_norm = self.predict(split_data["x_val_norm"])
        y_test_pred_norm = self.predict(split_data["x_test_norm"])
        y_train_pred = self.target_normalizer.inverse_transform(y_train_pred_norm)
        y_val_pred = self.target_normalizer.inverse_transform(y_val_pred_norm)
        y_test_pred = self.target_normalizer.inverse_transform(y_test_pred_norm)
        return {
            "split_data": split_data,
            "history": self.history,
            "train_metrics_norm": RegressionMetrics.evaluate(split_data["y_train_norm"], y_train_pred_norm, self.config.label_names),
            "val_metrics_norm": RegressionMetrics.evaluate(split_data["y_val_norm"], y_val_pred_norm, self.config.label_names),
            "test_metrics_norm": RegressionMetrics.evaluate(split_data["y_test_norm"], y_test_pred_norm, self.config.label_names),
            "train_metrics": RegressionMetrics.evaluate(split_data["y_train"], y_train_pred, self.config.label_names),
            "val_metrics": RegressionMetrics.evaluate(split_data["y_val"], y_val_pred, self.config.label_names),
            "test_metrics": RegressionMetrics.evaluate(split_data["y_test"], y_test_pred, self.config.label_names),
            "y_train_pred_norm": y_train_pred_norm,
            "y_val_pred_norm": y_val_pred_norm,
            "y_test_pred_norm": y_test_pred_norm,
            "y_train_pred": y_train_pred,
            "y_val_pred": y_val_pred,
            "y_test_pred": y_test_pred,
        }

    def save_artifacts(self, artifacts: dict[str, Any]) -> None:
        model_paths: dict[str, str] = {}
        best_iterations: dict[str, int] = {}
        best_scores: dict[str, float | None] = {}
        for label_name, model in self.models.items():
            model_path = self.model_dir / f"target_{label_name}.json"
            model.get_booster().save_model(model_path)
            model_paths[label_name] = str(model_path.name)
            best_iterations[label_name] = int(self.history["per_target"][label_name]["best_iteration"])
            raw_best_score = getattr(model, "best_score", None)
            best_scores[label_name] = None if raw_best_score is None else float(raw_best_score)

        model_metadata = {
            "config": asdict(self.config),
            "input_normalizer": self.input_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
            "input_feature_layout": {
                "stress_vector_features": self.config.input_stress_points,
                "appended_features": ["rate"],
                "total_features": self.config.input_size,
            },
            "model_paths": model_paths,
            "best_iterations": best_iterations,
            "best_scores": best_scores,
        }
        (self.model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="ascii")

        metrics_payload = {
            "config": asdict(self.config),
            "history": self.history,
            "input_normalizer": self.input_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
            "train_metrics_norm": artifacts["train_metrics_norm"],
            "val_metrics_norm": artifacts["val_metrics_norm"],
            "test_metrics_norm": artifacts["test_metrics_norm"],
            "train_metrics": artifacts["train_metrics"],
            "val_metrics": artifacts["val_metrics"],
            "test_metrics": artifacts["test_metrics"],
            "model_paths": model_paths,
            "best_iterations": best_iterations,
            "best_scores": best_scores,
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="ascii")
        np.savez(
            self.output_dir / "predictions.npz",
            y_train_pred=artifacts["y_train_pred"],
            y_val_pred=artifacts["y_val_pred"],
            y_test_pred=artifacts["y_test_pred"],
            y_train_pred_norm=artifacts["y_train_pred_norm"],
            y_val_pred_norm=artifacts["y_val_pred_norm"],
            y_test_pred_norm=artifacts["y_test_pred_norm"],
            y_train_true=artifacts["split_data"]["y_train"],
            y_val_true=artifacts["split_data"]["y_val"],
            y_test_true=artifacts["split_data"]["y_test"],
            rate_test=artifacts["split_data"]["rate_test"],
        )

    def save_plots(self, artifacts: dict[str, Any]) -> None:
        if plt is None:
            print("matplotlib is not installed; skipping plot generation.")
            return
        self._save_loss_plot()
        self._save_scatter_plot(artifacts["split_data"]["y_test"], artifacts["y_test_pred"], artifacts["test_metrics"]["r2"], "physical", False)
        self._save_scatter_plot(artifacts["split_data"]["y_test_norm"], artifacts["y_test_pred_norm"], artifacts["test_metrics_norm"]["r2"], "normalized", True)

    @staticmethod
    def require_pyplot() -> Any:
        if plt is None:
            raise RuntimeError("matplotlib is required for plot generation.")
        return plt

    def _save_loss_plot(self) -> None:
        if not self.history["train_loss"] or not self.history["val_loss"]:
            return
        pyplot = self.require_pyplot()
        rounds = np.arange(1, len(self.history["train_loss"]) + 1)
        fig, ax = pyplot.subplots(figsize=(8, 4.5))
        ax.plot(rounds, self.history["train_loss"], marker="o", linewidth=1.5, markersize=3, label="Train RMSE")
        ax.plot(rounds, self.history["val_loss"], marker="s", linewidth=1.5, markersize=3, label="Validation RMSE")
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("RMSE")
        ax.set_title("Mean RMSE Across Targets")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=160)
        pyplot.close(fig)

    def _save_scatter_plot(self, y_true: np.ndarray, y_pred: np.ndarray, r2_values: list[float], suffix: str, normalized: bool) -> None:
        pyplot = self.require_pyplot()
        fig, axes = pyplot.subplots(2, 2, figsize=(10, 8))
        axes = axes.reshape(-1)
        for idx, axis in enumerate(axes):
            axis.scatter(y_true[:, idx], y_pred[:, idx], s=14)
            if normalized:
                axis.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.2)
                axis.set_xlim(0.0, 1.0)
                axis.set_ylim(0.0, 1.0)
                axis.set_xlabel(f"{self.config.label_names[idx]} true (norm)")
                axis.set_ylabel(f"{self.config.label_names[idx]} pred (norm)")
            else:
                y_min = float(min(np.min(y_true[:, idx]), np.min(y_pred[:, idx])))
                y_max = float(max(np.max(y_true[:, idx]), np.max(y_pred[:, idx])))
                axis.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
                axis.set_xlabel(f"{self.config.label_names[idx]} true")
                axis.set_ylabel(f"{self.config.label_names[idx]} pred")
            axis.set_title(f"{self.config.label_names[idx]} | R^2 = {r2_values[idx]:.4f}")
            axis.grid(True)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"test_scatter_{suffix}.png", dpi=160)
        pyplot.close(fig)

    def print_summary(self, artifacts: dict[str, Any]) -> None:
        print("Normalized-space test MAE [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics_norm"]["mae"]))
        print("Normalized-space test RMSE [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics_norm"]["rmse"]))
        print("Normalized-space test R2 [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics_norm"]["r2"]))
        print("Test MAE [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics"]["mae"]))
        print("Test RMSE [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics"]["rmse"]))
        print("Test R2 [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics"]["r2"]))
        print("\nExample test predictions:")
        n_examples = min(5, artifacts["split_data"]["x_test"].shape[0])
        for idx in range(n_examples):
            rate = artifacts["split_data"]["rate_test"][idx]
            true_values = artifacts["split_data"]["y_test"][idx]
            pred_values = artifacts["y_test_pred"][idx]
            print(f"Sample {idx + 1} | rate = {rate:.2f} %/s")
            print(f"  true = [{true_values[0]:10.4f} {true_values[1]:10.4f} {true_values[2]:10.6f} {true_values[3]:10.4f}]")
            print(f"  pred = [{pred_values[0]:10.4f} {pred_values[1]:10.4f} {pred_values[2]:10.6f} {pred_values[3]:10.4f}]")
        print(f"\nSaved XGBoost model artifacts -> {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory name")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override the maximum boosting rounds")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="Override early stopping patience",
    )
    args = parser.parse_args()

    config = SMAXGBoostConfig()
    if args.output_dir is not None:
        config.output_dir_name = args.output_dir
    if args.n_estimators is not None:
        config.n_estimators = args.n_estimators
    if args.early_stopping_rounds is not None:
        config.early_stopping_rounds = args.early_stopping_rounds

    trainer = SMAXGBoostTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
