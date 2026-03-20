from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMAResidualDNNConfig(SMAAnnConfig):
    hidden_layers: tuple[int, ...] = (512, 256)
    dropout: float = 0.05
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 2.0
    loss_name: str = "huber"
    huber_delta: float = 0.025
    gradient_clip_norm: float | None = 1.0
    lr_scheduler_patience: int = 15
    early_stopping_patience: int = 40
    batch_size: int = 64
    validation_fraction: float = 0.15
    epochs: int = 220
    block_width: int = 256
    num_residual_blocks: int = 3


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


class SMAResidualDNNNet(nn.Module):
    def __init__(self, config: SMAResidualDNNConfig) -> None:
        super().__init__()
        if len(config.hidden_layers) < 2:
            raise ValueError("hidden_layers must contain at least 2 layers for residual DNN.")
        stem_hidden, trunk_width = config.hidden_layers[0], config.block_width
        self.stem = nn.Sequential(
            nn.Linear(config.input_size, stem_hidden),
            nn.LayerNorm(stem_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(stem_hidden, trunk_width),
            nn.LayerNorm(trunk_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(trunk_width, config.dropout) for _ in range(config.num_residual_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(trunk_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.output_size),
        )
        self.use_output_sigmoid = config.use_output_sigmoid
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        x = self.blocks(x)
        x = self.head(x)
        if self.use_output_sigmoid:
            x = self.output_sigmoid(x)
        return x


class SMAResidualDNNTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAResidualDNNConfig | None = None) -> None:
        resolved_config = config or SMAResidualDNNConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_resdnn_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAResidualDNNNet(self.config).to(self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual DNN regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--c-loss-weight", type=float, default=None, help="Override loss weight for coefficient C")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--num-residual-blocks",
        type=int,
        default=None,
        help="Override number of residual MLP blocks",
    )
    args = parser.parse_args()

    config = SMAResidualDNNConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.c_loss_weight is not None:
        config.c_loss_weight = args.c_loss_weight
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_residual_blocks is not None:
        config.num_residual_blocks = args.num_residual_blocks

    trainer = SMAResidualDNNTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
