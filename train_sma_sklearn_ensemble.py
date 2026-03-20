from __future__ import annotations

import argparse
import importlib
import json
import pickle
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
class SMASklearnEnsembleConfig(SMAAnnConfig):
    output_dir_name: str = "python_sklearn_outputs"
    estimator_name: str = "random_forest"
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | int | None = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1


class SMASklearnEnsembleTrainer:
    def __init__(self, root: Path, config: SMASklearnEnsembleConfig | None = None) -> None:
        self.root = root
        self.config = config or SMASklearnEnsembleConfig()
        self.loader = MatDatasetLoader(root, self.config)
        self.data_dir = self.loader.find_data_dir()
        self.output_dir = self.root / self.config.output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_normalizer = MinMaxNormalizer()
        self.target_normalizer = MinMaxNormalizer(
            np.asarray(self.config.target_min, dtype=np.float64),
            np.asarray(self.config.target_max, dtype=np.float64),
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.model: Any | None = None

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
        train_indices: list[int] = []
        val_indices: list[int] = []
        for idx, level in enumerate(rate_levels):
            idx_rate = np.where(np.abs(rate - level) < tol)[0]
            idx_rate = self.rng.permutation(idx_rate)
            n_train_rate = int(train_counts[idx])
            n_val_rate = int(val_counts[idx])
            n_needed = n_train_rate + n_val_rate
            if idx_rate.size < n_needed:
                raise ValueError(f"Not enough samples for rate {level:.4f}. Need {n_needed}, found {idx_rate.size}.")
            train_indices.extend(idx_rate[:n_train_rate].tolist())
            val_indices.extend(idx_rate[n_train_rate:n_needed].tolist())
        return (
            self.rng.permutation(np.asarray(train_indices, dtype=np.int64)),
            self.rng.permutation(np.asarray(val_indices, dtype=np.int64)),
        )

    @staticmethod
    def distribute_counts(total_count: int, n_bins: int) -> np.ndarray:
        base_count = total_count // n_bins
        counts = np.full(n_bins, base_count, dtype=np.int64)
        remainder = total_count - base_count * n_bins
        if remainder > 0:
            counts[:remainder] += 1
        return counts

    def build_estimator(self) -> Any:
        try:
            ensemble_module = importlib.import_module("sklearn.ensemble")
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required to run sklearn ensemble trainers. Install it with `pip install scikit-learn`."
            ) from exc

        estimator_map = {
            "random_forest": "RandomForestRegressor",
            "extra_trees": "ExtraTreesRegressor",
        }
        if self.config.estimator_name not in estimator_map:
            raise ValueError(f"Unsupported estimator_name '{self.config.estimator_name}'.")
        estimator_cls = getattr(ensemble_module, estimator_map[self.config.estimator_name])
        return estimator_cls(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.seed,
        )

    def train_model(self, split_data: dict[str, Any]) -> None:
        self.model = self.build_estimator()
        self.model.fit(split_data["x_train_norm"], split_data["y_train_norm"])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained.")
        prediction = self.model.predict(x)
        return np.asarray(prediction, dtype=np.float32)

    def evaluate(self, split_data: dict[str, Any]) -> dict[str, Any]:
        y_train_pred_norm = self.predict(split_data["x_train_norm"])
        y_val_pred_norm = self.predict(split_data["x_val_norm"])
        y_test_pred_norm = self.predict(split_data["x_test_norm"])
        y_train_pred = self.target_normalizer.inverse_transform(y_train_pred_norm)
        y_val_pred = self.target_normalizer.inverse_transform(y_val_pred_norm)
        y_test_pred = self.target_normalizer.inverse_transform(y_test_pred_norm)
        return {
            "split_data": split_data,
            "history": {"train_loss": [], "val_loss": []},
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
        if self.model is None:
            raise ValueError("Model is not trained.")
        with (self.output_dir / "sma_sklearn_regressor.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "config": asdict(self.config),
                    "input_normalizer": self.input_normalizer.to_dict(),
                    "target_normalizer": self.target_normalizer.to_dict(),
                },
                handle,
            )
        metrics_payload = {
            "config": asdict(self.config),
            "history": artifacts["history"],
            "input_normalizer": self.input_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
            "train_metrics_norm": artifacts["train_metrics_norm"],
            "val_metrics_norm": artifacts["val_metrics_norm"],
            "test_metrics_norm": artifacts["test_metrics_norm"],
            "train_metrics": artifacts["train_metrics"],
            "val_metrics": artifacts["val_metrics"],
            "test_metrics": artifacts["test_metrics"],
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
        self._save_scatter_plot(artifacts["split_data"]["y_test"], artifacts["y_test_pred"], artifacts["test_metrics"]["r2"], "physical", False)
        self._save_scatter_plot(artifacts["split_data"]["y_test_norm"], artifacts["y_test_pred_norm"], artifacts["test_metrics_norm"]["r2"], "normalized", True)

    def _save_scatter_plot(self, y_true: np.ndarray, y_pred: np.ndarray, r2_values: list[float], suffix: str, normalized: bool) -> None:
        if plt is None:
            return
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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
        plt.close(fig)

    def print_summary(self, artifacts: dict[str, Any]) -> None:
        print(f"Estimator = {self.config.estimator_name}")
        print("Normalized-space test R2 [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics_norm"]["r2"]))
        print("Test R2 [C, L, k, Asd] =")
        print(np.array(artifacts["test_metrics"]["r2"]))
        print(f"\nSaved sklearn ensemble artifacts -> {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn ensemble regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument(
        "--estimator-name",
        choices=("random_forest", "extra_trees"),
        default="random_forest",
        help="Ensemble estimator to train",
    )
    args = parser.parse_args()

    config = SMASklearnEnsembleConfig(estimator_name=args.estimator_name)
    trainer = SMASklearnEnsembleTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
