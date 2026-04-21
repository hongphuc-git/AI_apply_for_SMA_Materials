from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_sma_ann import (
    MatDatasetLoader,
    MinMaxNormalizer,
    RegressionMetrics,
    SMAAnnConfig,
    SMAAnnTrainer,
    WeightedHuberLoss,
    WeightedMSELoss,
    create_torch_optimizer,
    estimate_runtime_band,
    summarize_split_shapes,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


@dataclass
class SMAHybridSplitConfig(SMAAnnConfig):
    output_dir_name: str = "python_hybrid_split_outputs"
    loading_points: int = 200
    unloading_points: int = 200
    xgb_n_estimators: int = 1200
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 6
    xgb_min_child_weight: float = 2.0
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_gamma: float = 0.0
    xgb_early_stopping_rounds: int = 80
    xgb_n_jobs: int = -1
    asd_hidden_layers: tuple[int, ...] = (512, 256)
    asd_dropout: float = 0.05
    asd_learning_rate: float = 3e-4
    asd_weight_decay: float = 1e-5
    asd_batch_size: int = 64
    asd_epochs: int = 180
    asd_block_width: int = 256
    asd_num_residual_blocks: int = 5
    loss_name: str = "huber"
    huber_delta: float = 0.02
    early_stopping_patience: int = 25
    gradient_clip_norm: float | None = 1.0


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        x = self.fc1(features)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        x = self.dropout(x)
        return x


class AsdResidualNet(nn.Module):
    def __init__(self, config: SMAHybridSplitConfig) -> None:
        super().__init__()
        if len(config.asd_hidden_layers) < 2:
            raise ValueError("asd_hidden_layers must contain at least 2 layers.")
        stem_hidden = config.asd_hidden_layers[0]
        trunk_width = config.asd_block_width
        self.stem = nn.Sequential(
            nn.Linear(config.unloading_points + 1, stem_hidden),
            nn.LayerNorm(stem_hidden),
            nn.GELU(),
            nn.Dropout(config.asd_dropout),
            nn.Linear(stem_hidden, config.asd_hidden_layers[1]),
            nn.LayerNorm(config.asd_hidden_layers[1]),
            nn.GELU(),
            nn.Dropout(config.asd_dropout),
            nn.Linear(config.asd_hidden_layers[1], trunk_width),
            nn.LayerNorm(trunk_width),
            nn.GELU(),
            nn.Dropout(config.asd_dropout),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(trunk_width, config.asd_dropout) for _ in range(config.asd_num_residual_blocks)])
        self.head = nn.Sequential(
            nn.Linear(trunk_width, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config.asd_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        x = self.blocks(x)
        return self.head(x)


class SMAHybridSplitTrainer:
    def __init__(self, root: Path, config: SMAHybridSplitConfig | None = None) -> None:
        self.root = root
        self.config = config or SMAHybridSplitConfig()
        self.loader = MatDatasetLoader(root, self.config)
        self.data_dir = self.loader.find_data_dir()
        self.output_dir = self.root / self.config.output_dir_name
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir = self.output_dir / "sma_hybrid_split_regressor"
        self.model_dir.mkdir(exist_ok=True)
        self.input_normalizer_clk = MinMaxNormalizer()
        self.input_normalizer_asd = MinMaxNormalizer()
        self.target_normalizer = MinMaxNormalizer(
            np.asarray(self.config.target_min, dtype=np.float64),
            np.asarray(self.config.target_max, dtype=np.float64),
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xgb_models: dict[str, Any] = {}
        self.asd_model = AsdResidualNet(self.config).to(self.device)
        self.history = {"train_loss": [], "val_loss": [], "per_target": {}, "asd_train_loss": [], "asd_val_loss": []}
        self.run_started_at: float | None = None

    def run(self) -> None:
        train_raw = self.loader.load_split(self.data_dir / "train.mat")
        test_raw = self.loader.load_split(self.data_dir / "test.mat")
        split_data = self.prepare_datasets(train_raw, test_raw)
        self.print_run_overview(split_data)
        self.run_started_at = time.perf_counter()
        self.train_model(split_data)
        artifacts = self.evaluate(split_data)
        self.save_artifacts(artifacts)
        self.save_plots(artifacts)
        self.print_summary(artifacts)

    def print_run_overview(self, split_data: dict[str, Any]) -> None:
        shape_summary = summarize_split_shapes(split_data)
        print("Run overview")
        print(f"  Data directory      : {self.data_dir}")
        print("  Data files         : train.mat + test.mat")
        print(f"  Input format       : loading[200]+rate -> XGBoost for C/L/k; unloading[200]+rate -> residual net for Asd")
        print(f"  Target format      : Y -> {shape_summary['target_dim']} regression targets {self.config.label_names}")
        print(f"  Dataset split      : train={shape_summary['train_samples']}, val={shape_summary['val_samples']}, test={shape_summary['test_samples']}")
        print(f"  Rate coverage      : {shape_summary['rate_summary']}")
        print(f"  Device             : {'GPU' if self.device.type == 'cuda' else 'CPU'} for Asd branch | XGBoost CPU/host")
        print("  Model              : HybridSplit (XGBoost C/L/k + ResidualNet Asd)")
        print(f"  Config             : xgb_estimators={self.config.xgb_n_estimators}, asd_epochs={self.config.asd_epochs}, asd_batch={self.config.asd_batch_size}")

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

        load_train = x_train[:, : self.config.loading_points]
        load_val = x_val[:, : self.config.loading_points]
        load_test = x_test[:, : self.config.loading_points]
        unload_train = x_train[:, self.config.loading_points :]
        unload_val = x_val[:, self.config.loading_points :]
        unload_test = x_test[:, self.config.loading_points :]

        x_clk_train = np.column_stack([load_train, rate_train]).astype(np.float32)
        x_clk_val = np.column_stack([load_val, rate_val]).astype(np.float32)
        x_clk_test = np.column_stack([load_test, rate_test]).astype(np.float32)
        x_asd_train = np.column_stack([unload_train, rate_train]).astype(np.float32)
        x_asd_val = np.column_stack([unload_val, rate_val]).astype(np.float32)
        x_asd_test = np.column_stack([unload_test, rate_test]).astype(np.float32)

        x_clk_train_norm = self.input_normalizer_clk.fit(x_clk_train).transform(x_clk_train)
        x_clk_val_norm = self.input_normalizer_clk.transform(x_clk_val)
        x_clk_test_norm = self.input_normalizer_clk.transform(x_clk_test)
        x_asd_train_norm = self.input_normalizer_asd.fit(x_asd_train).transform(x_asd_train)
        x_asd_val_norm = self.input_normalizer_asd.transform(x_asd_val)
        x_asd_test_norm = self.input_normalizer_asd.transform(x_asd_test)

        y_train_norm = self.target_normalizer.transform(y_train).astype(np.float32)
        y_val_norm = self.target_normalizer.transform(y_val).astype(np.float32)
        y_test_norm = self.target_normalizer.transform(y_test).astype(np.float32)

        return {
            "epsTarget": train_raw["epsTarget"],
            "x_train": np.column_stack([x_train, rate_train]).astype(np.float32),
            "y_train": y_train,
            "rate_train": rate_train,
            "x_val": np.column_stack([x_val, rate_val]).astype(np.float32),
            "y_val": y_val,
            "rate_val": rate_val,
            "x_test": np.column_stack([x_test, rate_test]).astype(np.float32),
            "y_test": y_test,
            "rate_test": rate_test,
            "y_train_norm": y_train_norm,
            "y_val_norm": y_val_norm,
            "y_test_norm": y_test_norm,
            "x_train_norm": np.column_stack([x_train, rate_train]).astype(np.float32),
            "x_val_norm": np.column_stack([x_val, rate_val]).astype(np.float32),
            "x_test_norm": np.column_stack([x_test, rate_test]).astype(np.float32),
            "x_clk_train_norm": x_clk_train_norm,
            "x_clk_val_norm": x_clk_val_norm,
            "x_clk_test_norm": x_clk_test_norm,
            "x_asd_train_norm": x_asd_train_norm,
            "x_asd_val_norm": x_asd_val_norm,
            "x_asd_test_norm": x_asd_test_norm,
        }

    def stratified_split(self, rate: np.ndarray, train_target: int, val_target: int) -> tuple[np.ndarray, np.ndarray]:
        rate_levels = np.unique(rate)
        total_count = train_target + val_target
        val_fraction = val_target / max(total_count, 1)
        tol = 1e-6
        train_index_list: list[int] = []
        val_index_list: list[int] = []
        for idx, level in enumerate(rate_levels):
            idx_rate = np.where(np.abs(rate - level) < tol)[0]
            idx_rate = self.rng.permutation(idx_rate)
            n_val_rate = int(round(val_fraction * idx_rate.size))
            n_val_rate = max(1, min(idx_rate.size - 1, n_val_rate))
            n_train_rate = idx_rate.size - n_val_rate
            n_needed = n_train_rate + n_val_rate
            if idx_rate.size < n_needed:
                raise ValueError(f"Not enough samples for rate {level:.4f}. Need {n_needed}, found {idx_rate.size}.")
            train_index_list.extend(idx_rate[:n_train_rate].tolist())
            val_index_list.extend(idx_rate[n_train_rate:n_needed].tolist())
        train_indices = self.rng.permutation(np.asarray(train_index_list, dtype=np.int64))
        val_indices = self.rng.permutation(np.asarray(val_index_list, dtype=np.int64))
        return train_indices, val_indices

    def build_xgb_model(self) -> Any:
        try:
            xgboost_module = importlib.import_module("xgboost")
        except ImportError as exc:
            raise ImportError("xgboost is required for hybrid split trainer. Install it with `pip install xgboost`.") from exc
        xgb_regressor_cls = getattr(xgboost_module, "XGBRegressor")
        return xgb_regressor_cls(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            min_child_weight=self.config.xgb_min_child_weight,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            gamma=self.config.xgb_gamma,
            early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            random_state=self.config.seed,
            n_jobs=self.config.xgb_n_jobs,
        )

    def train_model(self, split_data: dict[str, Any]) -> None:
        for target_idx, label_name in enumerate(("C", "L", "k")):
            model = self.build_xgb_model()
            y_train_target = split_data["y_train_norm"][:, target_idx]
            y_val_target = split_data["y_val_norm"][:, target_idx]
            model.fit(
                split_data["x_clk_train_norm"],
                y_train_target,
                eval_set=[
                    (split_data["x_clk_train_norm"], y_train_target),
                    (split_data["x_clk_val_norm"], y_val_target),
                ],
                verbose=False,
            )
            self.xgb_models[label_name] = model
            evals_result = model.evals_result()
            train_curve = [float(v) for v in evals_result.get("validation_0", {}).get("rmse", [])]
            val_curve = [float(v) for v in evals_result.get("validation_1", {}).get("rmse", [])]
            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is None:
                best_iteration = max(len(val_curve), 1) - 1
            self.history["per_target"][label_name] = {"train_rmse": train_curve, "val_rmse": val_curve, "best_iteration": int(best_iteration)}

        train_loader = DataLoader(TensorDataset(torch.from_numpy(split_data["x_asd_train_norm"]).float(), torch.from_numpy(split_data["y_train_norm"][:, 3:4]).float()), batch_size=self.config.asd_batch_size, shuffle=True)
        criterion: nn.Module
        if self.config.loss_name == "huber":
            criterion = WeightedHuberLoss(torch.ones(1, dtype=torch.float32, device=self.device), self.config.huber_delta)
        else:
            criterion = WeightedMSELoss(torch.ones(1, dtype=torch.float32, device=self.device))
        optimizer = create_torch_optimizer(self.asd_model.parameters(), self.config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.config.lr_scheduler_factor, patience=self.config.lr_scheduler_patience)
        best_val = float("inf")
        best_state = None
        patience = 0

        x_val = torch.from_numpy(split_data["x_asd_val_norm"]).float().to(self.device)
        y_val = torch.from_numpy(split_data["y_val_norm"][:, 3:4]).float().to(self.device)
        for epoch in range(self.config.asd_epochs):
            self.asd_model.train()
            total_loss = 0.0
            total_count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.asd_model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.asd_model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                total_count += xb.size(0)
            train_loss = total_loss / max(total_count, 1)
            self.asd_model.eval()
            with torch.no_grad():
                val_pred = self.asd_model(x_val)
                val_loss = float(criterion(val_pred, y_val).item())
            self.history["asd_train_loss"].append(train_loss)
            self.history["asd_val_loss"].append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.asd_model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break
        if best_state is not None:
            self.asd_model.load_state_dict(best_state)

        self.history["train_loss"] = self.aggregate_xgb_history("train_rmse")
        self.history["val_loss"] = self.aggregate_xgb_history("val_rmse")

    def aggregate_xgb_history(self, key: str) -> list[float]:
        curves = [np.asarray(h[key], dtype=np.float64) for h in self.history["per_target"].values() if key in h and h[key]]
        if not curves:
            return []
        max_len = max(c.size for c in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for i, c in enumerate(curves):
            padded[i, : c.size] = c
        return np.nanmean(padded, axis=0).tolist()

    def predict_norm(self, split_data: dict[str, Any], split: str) -> np.ndarray:
        x_clk = split_data[f"x_clk_{split}_norm"]
        x_asd = split_data[f"x_asd_{split}_norm"]
        clk_cols = []
        for label_name in ("C", "L", "k"):
            model = self.xgb_models[label_name]
            clk_cols.append(np.asarray(model.predict(x_clk), dtype=np.float32).reshape(-1, 1))
        clk_pred = np.concatenate(clk_cols, axis=1)
        self.asd_model.eval()
        with torch.no_grad():
            asd_pred = self.asd_model(torch.from_numpy(x_asd).float().to(self.device)).cpu().numpy().astype(np.float32)
        pred = np.concatenate([clk_pred, asd_pred], axis=1)
        return SMAAnnTrainer.constrain_normalized_predictions(pred)

    def evaluate(self, split_data: dict[str, Any]) -> dict[str, Any]:
        y_train_pred_norm = self.predict_norm(split_data, "train")
        y_val_pred_norm = self.predict_norm(split_data, "val")
        y_test_pred_norm = self.predict_norm(split_data, "test")
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
        for label_name, model in self.xgb_models.items():
            model_path = self.model_dir / f"target_{label_name}.json"
            model.get_booster().save_model(model_path)
            model_paths[label_name] = str(model_path.name)
            best_iterations[label_name] = int(self.history["per_target"][label_name]["best_iteration"])
        torch.save(
            {
                "model_state_dict": self.asd_model.state_dict(),
                "config": asdict(self.config),
                "input_normalizer_clk": self.input_normalizer_clk.to_dict(),
                "input_normalizer_asd": self.input_normalizer_asd.to_dict(),
                "target_normalizer": self.target_normalizer.to_dict(),
                "xgb_model_paths": model_paths,
                "xgb_best_iterations": best_iterations,
            },
            self.output_dir / "sma_hybrid_split_regressor.pt",
        )
        metrics_payload = {
            "config": asdict(self.config),
            "history": self.history,
            "train_metrics_norm": artifacts["train_metrics_norm"],
            "val_metrics_norm": artifacts["val_metrics_norm"],
            "test_metrics_norm": artifacts["test_metrics_norm"],
            "train_metrics": artifacts["train_metrics"],
            "val_metrics": artifacts["val_metrics"],
            "test_metrics": artifacts["test_metrics"],
            "xgb_model_paths": model_paths,
            "xgb_best_iterations": best_iterations,
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="ascii")

    def save_plots(self, artifacts: dict[str, Any]) -> None:
        if plt is None:
            print("matplotlib is not installed; skipping plot generation.")
            return
        epochs = np.arange(1, len(self.history["asd_train_loss"]) + 1)
        if epochs.size:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(epochs, self.history["asd_train_loss"], marker="o", linewidth=1.5, markersize=3, label="Asd Train")
            ax.plot(epochs, self.history["asd_val_loss"], marker="s", linewidth=1.5, markersize=3, label="Asd Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Asd Branch Training and Validation Loss")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / "loss_curve.png", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.reshape(-1)
        y_true = artifacts["split_data"]["y_test"]
        y_pred = artifacts["y_test_pred"]
        r2_values = artifacts["test_metrics"]["r2"]
        for idx, axis in enumerate(axes):
            axis.scatter(y_true[:, idx], y_pred[:, idx], s=14)
            y_min = float(min(np.min(y_true[:, idx]), np.min(y_pred[:, idx])))
            y_max = float(max(np.max(y_true[:, idx]), np.max(y_pred[:, idx])))
            axis.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
            axis.set_xlabel(f"{self.config.label_names[idx]} true")
            axis.set_ylabel(f"{self.config.label_names[idx]} pred")
            axis.set_title(f"{self.config.label_names[idx]} | R^2 = {r2_values[idx]:.4f}")
            axis.grid(True)
        fig.tight_layout()
        fig.savefig(self.output_dir / "test_scatter_physical.png", dpi=160)
        plt.close(fig)

    def print_summary(self, artifacts: dict[str, Any]) -> None:
        if self.run_started_at is not None:
            elapsed_seconds = time.perf_counter() - self.run_started_at
            print(f"Elapsed time = {elapsed_seconds / 60.0:.2f} minutes")
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
        print(f"\nSaved hybrid model artifacts -> {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid split SMA regressor: XGBoost for C/L/k and residual net for Asd.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override Asd epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Asd batch size")
    args = parser.parse_args()
    config = SMAHybridSplitConfig()
    if args.epochs is not None:
        config.asd_epochs = args.epochs
    if args.batch_size is not None:
        config.asd_batch_size = args.batch_size
    trainer = SMAHybridSplitTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
