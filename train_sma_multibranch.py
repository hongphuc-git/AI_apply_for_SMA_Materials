from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMAMultiBranchConfig(SMAAnnConfig):
    loading_points: int = 200
    unloading_points: int = 200
    loading_hidden: tuple[int, ...] = (256, 128)
    unloading_hidden: tuple[int, ...] = (256, 128)
    rate_hidden: tuple[int, ...] = (32, 16)
    fusion_hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.06
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 2.5
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


class SMAMultiBranchNet(nn.Module):
    def __init__(self, config: SMAMultiBranchConfig) -> None:
        super().__init__()
        if config.loading_points + config.unloading_points != config.input_stress_points:
            raise ValueError("loading_points + unloading_points must equal input_stress_points.")

        self.loading_points = config.loading_points
        self.unloading_points = config.unloading_points

        self.loading_branch = make_mlp(config.loading_points, config.loading_hidden, config.dropout)
        self.unloading_branch = make_mlp(config.unloading_points, config.unloading_hidden, config.dropout)
        self.rate_branch = make_mlp(1, config.rate_hidden, config.dropout)

        loading_out = config.loading_hidden[-1] if config.loading_hidden else config.loading_points
        unloading_out = config.unloading_hidden[-1] if config.unloading_hidden else config.unloading_points
        rate_out = config.rate_hidden[-1] if config.rate_hidden else 1
        fusion_in = loading_out + unloading_out + rate_out

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
        fusion_layers.append(nn.Linear(current, config.output_size))
        if config.use_output_sigmoid:
            fusion_layers.append(nn.Sigmoid())
        self.fusion_head = nn.Sequential(*fusion_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stress = features[:, :-1]
        rate = features[:, -1:]
        loading = stress[:, : self.loading_points]
        unloading = stress[:, self.loading_points : self.loading_points + self.unloading_points]

        loading_repr = self.loading_branch(loading)
        unloading_repr = self.unloading_branch(unloading)
        rate_repr = self.rate_branch(rate)
        fused = torch.cat([loading_repr, unloading_repr, rate_repr], dim=1)
        return self.fusion_head(fused)


class SMAMultiBranchTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAMultiBranchConfig | None = None) -> None:
        resolved_config = config or SMAMultiBranchConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_multibranch_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAMultiBranchNet(self.config).to(self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-branch SMA regressor for loading/unloading/rate fusion.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = SMAMultiBranchConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    trainer = SMAMultiBranchTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
