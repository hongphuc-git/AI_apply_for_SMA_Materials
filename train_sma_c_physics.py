from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMACPhysicsConfig(SMAAnnConfig):
    output_size: int = 4
    label_names: tuple[str, ...] = ("C",)
    target_min: tuple[float, ...] = (800.0,)
    target_max: tuple[float, ...] = (2000.0,)
    loading_points: int = 200
    unloading_points: int = 200
    loading_hidden: tuple[int, ...] = (256, 128)
    hysteresis_hidden: tuple[int, ...] = (128, 64)
    rate_hidden: tuple[int, ...] = (48, 24)
    thermal_hidden: tuple[int, ...] = (96, 48)
    fusion_hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.05
    learning_rate: float = 4e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 1.0
    loss_name: str = "huber"
    huber_delta: float = 0.02
    batch_size: int = 64
    epochs: int = 220
    validation_fraction: float = 0.15


def make_mlp(in_features: int, hidden_layers: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_features
    for width in hidden_layers:
        layers.extend(
            [
                nn.Linear(current, width),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        current = width
    return nn.Sequential(*layers)


class SMACPhysicsNet(nn.Module):
    def __init__(self, config: SMACPhysicsConfig) -> None:
        super().__init__()
        if config.loading_points + config.unloading_points != config.input_stress_points:
            raise ValueError("loading_points + unloading_points must equal input_stress_points.")

        self.loading_points = config.loading_points
        self.unloading_points = config.unloading_points

        self.loading_branch = make_mlp(config.loading_points, config.loading_hidden, config.dropout)
        self.hysteresis_branch = make_mlp(config.loading_points, config.hysteresis_hidden, config.dropout)
        self.rate_branch = make_mlp(6, config.rate_hidden, config.dropout)
        self.thermal_branch = make_mlp(6, config.thermal_hidden, config.dropout)

        loading_out = config.loading_hidden[-1] if config.loading_hidden else config.loading_points
        hysteresis_out = config.hysteresis_hidden[-1] if config.hysteresis_hidden else config.loading_points
        rate_out = config.rate_hidden[-1] if config.rate_hidden else 6
        thermal_out = config.thermal_hidden[-1] if config.thermal_hidden else 6
        fusion_in = loading_out + hysteresis_out + rate_out + thermal_out

        fusion_layers: list[nn.Module] = []
        current = fusion_in
        for width in config.fusion_hidden:
            fusion_layers.extend(
                [
                    nn.Linear(current, width),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            current = width
        fusion_layers.append(nn.Linear(current, 1))
        if config.use_output_sigmoid:
            fusion_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*fusion_layers)

    @staticmethod
    def build_rate_features(loading: torch.Tensor, unloading: torch.Tensor, rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unloading_reversed = torch.flip(unloading, dims=[1])
        gap = torch.abs(loading - unloading_reversed)
        peak_stress = torch.max(torch.cat([loading, unloading], dim=1), dim=1, keepdim=True).values
        loop_area_proxy = gap.mean(dim=1, keepdim=True)
        mean_loading = (loading[:, 1:] - loading[:, :-1]).mean(dim=1, keepdim=True)
        mean_unloading = (unloading[:, :-1] - unloading[:, 1:]).mean(dim=1, keepdim=True)
        mid_idx = loading.size(1) // 2
        mid_gap = torch.abs(loading[:, mid_idx : mid_idx + 1] - unloading_reversed[:, mid_idx : mid_idx + 1])
        thermal_rate_proxy = loop_area_proxy * rate
        return torch.cat([rate, peak_stress, loop_area_proxy, mean_loading, mean_unloading, mid_gap], dim=1), torch.cat(
            [peak_stress, loop_area_proxy, thermal_rate_proxy, mean_loading * rate, mean_unloading * rate, mid_gap * rate], dim=1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stress = features[:, :-1]
        rate = features[:, -1:]
        loading = stress[:, : self.loading_points]
        unloading = stress[:, self.loading_points : self.loading_points + self.unloading_points]
        unloading_reversed = torch.flip(unloading, dims=[1])
        hysteresis_signal = loading - unloading_reversed
        rate_features, thermal_features = self.build_rate_features(loading, unloading, rate)

        loading_repr = self.loading_branch(loading)
        hysteresis_repr = self.hysteresis_branch(hysteresis_signal)
        rate_repr = self.rate_branch(rate_features)
        thermal_repr = self.thermal_branch(thermal_features)
        fused = torch.cat([loading_repr, hysteresis_repr, rate_repr, thermal_repr], dim=1)
        return self.head(fused)


class SMACPhysicsTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMACPhysicsConfig | None = None) -> None:
        resolved_config = config or SMACPhysicsConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_c_physics_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMACPhysicsNet(self.config).to(self.device)

    def make_weighted_mse(self) -> nn.Module:
        weights = torch.ones(1, dtype=torch.float32, device=self.device)
        if self.config.loss_name == "mse":
            from train_sma_ann import WeightedMSELoss

            return WeightedMSELoss(weights)
        if self.config.loss_name == "huber":
            from train_sma_ann import WeightedHuberLoss

            return WeightedHuberLoss(weights, self.config.huber_delta)
        raise ValueError(f"Unsupported loss_name '{self.config.loss_name}'.")

    def save_plots(self, artifacts: dict[str, Any]) -> None:
        self._save_loss_plot()
        self._save_single_scatter_plot(
            artifacts["split_data"]["y_test"],
            artifacts["y_test_pred"],
            float(artifacts["test_metrics"]["r2"][0]),
            "physical",
            False,
        )
        self._save_single_scatter_plot(
            artifacts["split_data"]["y_test_norm"],
            artifacts["y_test_pred_norm"],
            float(artifacts["test_metrics_norm"]["r2"][0]),
            "normalized",
            True,
        )

    def _save_single_scatter_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        r2_value: float,
        suffix: str,
        normalized: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(y_true[:, 0], y_pred[:, 0], s=14)
        if normalized:
            ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.2)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("C true (norm)")
            ax.set_ylabel("C pred (norm)")
        else:
            y_min = float(min(np.min(y_true[:, 0]), np.min(y_pred[:, 0])))
            y_max = float(max(np.max(y_true[:, 0]), np.max(y_pred[:, 0])))
            ax.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
            ax.set_xlabel("C true")
            ax.set_ylabel("C pred")
        ax.set_title(f"C | R^2 = {r2_value:.4f}")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"test_scatter_{suffix}.png", dpi=160)
        plt.close(fig)

    def print_summary(self, artifacts: dict[str, Any]) -> None:
        import time

        if self.run_started_at is not None:
            elapsed_seconds = time.perf_counter() - self.run_started_at
            print(f"Elapsed time = {elapsed_seconds / 60.0:.2f} minutes")
        print("Normalized-space test MAE [C] =")
        print(np.array(artifacts["test_metrics_norm"]["mae"]))
        print("Normalized-space test RMSE [C] =")
        print(np.array(artifacts["test_metrics_norm"]["rmse"]))
        print("Normalized-space test R2 [C] =")
        print(np.array(artifacts["test_metrics_norm"]["r2"]))
        print("Test MAE [C] =")
        print(np.array(artifacts["test_metrics"]["mae"]))
        print("Test RMSE [C] =")
        print(np.array(artifacts["test_metrics"]["rmse"]))
        print("Test R2 [C] =")
        print(np.array(artifacts["test_metrics"]["r2"]))

    def prepare_datasets(self, train_raw: dict[str, np.ndarray], test_raw: dict[str, np.ndarray]) -> dict[str, Any]:
        split_data = super().prepare_datasets(train_raw, test_raw)
        for key in ("y_train", "y_val", "y_test", "y_train_norm", "y_val_norm", "y_test_norm"):
            split_data[key] = np.asarray(split_data[key][:, :1], dtype=np.float32)
        return split_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C-only physics-guided SMA regressor.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = SMACPhysicsConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    trainer = SMACPhysicsTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
