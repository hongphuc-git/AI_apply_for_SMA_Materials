from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMACNNConfig(SMAAnnConfig):
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (7, 5, 3)
    cnn_dropout: float = 0.10
    hidden_layers: tuple[int, ...] = (256, 128)
    dropout: float = 0.10
    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 3.0
    epochs: int = 180


class SMA1DCNNNet(nn.Module):
    def __init__(self, config: SMACNNConfig) -> None:
        super().__init__()
        if len(config.conv_channels) != len(config.kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel_size in zip(config.conv_channels, config.kernel_sizes):
            padding = kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(config.cnn_dropout),
                ]
            )
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        mlp_layers: list[nn.Module] = []
        in_features = config.conv_channels[-1] + 1
        for hidden_size in config.hidden_layers:
            mlp_layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_features = hidden_size
        mlp_layers.append(nn.Linear(in_features, config.output_size))
        if config.use_output_sigmoid:
            mlp_layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*mlp_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stress_curve = features[:, :-1].unsqueeze(1)
        rate = features[:, -1:].contiguous()
        x = self.feature_extractor(stress_curve)
        x = self.global_pool(x).squeeze(-1)
        x = torch.cat([x, rate], dim=1)
        return self.regressor(x)


class SMACNNTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMACNNConfig | None = None) -> None:
        resolved_config = config or SMACNNConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_cnn_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMA1DCNNNet(self.config).to(self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D CNN regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--c-loss-weight", type=float, default=None, help="Override loss weight for coefficient C")
    args = parser.parse_args()

    config = SMACNNConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.c_loss_weight is not None:
        config.c_loss_weight = args.c_loss_weight

    trainer = SMACNNTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
