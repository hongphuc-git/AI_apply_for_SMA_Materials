from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer, create_torch_optimizer


@dataclass
class SMAResidualDNNV2Config(SMAAnnConfig):
    hidden_layers: tuple[int, ...] = (768, 512)
    dropout: float = 0.03
    learning_rate: float = 1.5e-4
    weight_decay: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    c_loss_weight: float = 2.4
    loss_name: str = "huber"
    huber_delta: float = 0.02
    gradient_clip_norm: float | None = 0.8
    batch_size: int = 96
    validation_fraction: float = 0.18
    epochs: int = 260
    lr_scheduler_patience: int = 20
    early_stopping_patience: int = 55
    block_width: int = 384
    num_residual_blocks: int = 6
    layer_scale_init: float = 0.10
    warmup_epochs: int = 12
    min_learning_rate: float = 1e-5


class StableResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float, layer_scale_init: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), float(layer_scale_init)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm1(features)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return features + x * self.layer_scale


class SMAStableResidualDNNNet(nn.Module):
    def __init__(self, config: SMAResidualDNNV2Config) -> None:
        super().__init__()
        if len(config.hidden_layers) < 2:
            raise ValueError("hidden_layers must contain at least 2 layers for stable residual DNN.")
        stem_hidden, trunk_width = config.hidden_layers[0], config.block_width
        self.stem = nn.Sequential(
            nn.Linear(config.input_size, stem_hidden),
            nn.LayerNorm(stem_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(stem_hidden, trunk_width),
            nn.LayerNorm(trunk_width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                StableResidualBlock(
                    trunk_width,
                    dropout=config.dropout,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(trunk_width),
            nn.Linear(trunk_width, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
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


class SMAResidualDNNV2Trainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAResidualDNNV2Config | None = None) -> None:
        resolved_config = config or SMAResidualDNNV2Config()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_resdnn_v2_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAStableResidualDNNNet(self.config).to(self.device)

    def train_model(self, split_data: dict[str, Any]) -> None:
        train_loader = self.make_loader(split_data["x_train_norm"], split_data["y_train_norm"], shuffle=True)
        criterion = self.make_weighted_mse()
        optimizer = create_torch_optimizer(self.model.parameters(), self.config)
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        total_epochs = max(self.config.epochs, 1)
        warmup_epochs = max(1, min(self.config.warmup_epochs, total_epochs))

        for epoch in range(1, total_epochs + 1):
            current_lr = self.compute_epoch_lr(epoch, total_epochs, warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            self.model.train()
            train_loss_sum = 0.0
            train_count = 0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                if self.config.gradient_clip_norm is not None and self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                batch_size = features.size(0)
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss = self.compute_loss(split_data["x_val_norm"], split_data["y_val_norm"], criterion)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss + self.config.early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:3d}/{total_epochs:3d} | "
                f"train_loss={train_loss:.6e} | val_loss={val_loss:.6e} | lr={current_lr:.2e}"
            )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"after {epoch} epochs (best val_loss={best_val_loss:.6e})."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def compute_epoch_lr(self, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
        base_lr = self.config.learning_rate
        min_lr = self.config.min_learning_rate
        if epoch <= warmup_epochs:
            return base_lr * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stable deeper residual DNN regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--c-loss-weight", type=float, default=None, help="Override loss weight for coefficient C")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-residual-blocks", type=int, default=None, help="Override number of residual blocks")
    parser.add_argument("--block-width", type=int, default=None, help="Override residual block width")
    args = parser.parse_args()

    config = SMAResidualDNNV2Config()
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
    if args.block_width is not None:
        config.block_width = args.block_width

    trainer = SMAResidualDNNV2Trainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
