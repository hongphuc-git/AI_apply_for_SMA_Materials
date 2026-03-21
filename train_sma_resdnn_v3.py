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
class SMAResidualDNNV3Config(SMAAnnConfig):
    hidden_layers: tuple[int, ...] = (768, 512)
    dropout: float = 0.025
    learning_rate: float = 1.2e-4
    weight_decay: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    c_loss_weight: float = 2.8
    loss_name: str = "huber"
    huber_delta: float = 0.018
    gradient_clip_norm: float | None = 0.7
    batch_size: int = 128
    validation_fraction: float = 0.20
    epochs: int = 300
    early_stopping_patience: int = 65
    block_width: int = 384
    num_residual_blocks: int = 8
    layer_scale_init: float = 0.08
    warmup_epochs: int = 14
    min_learning_rate: float = 8e-6
    ema_decay: float = 0.995


class StableResidualBlockV3(nn.Module):
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


class SMAStableResidualDNNV3Net(nn.Module):
    def __init__(self, config: SMAResidualDNNV3Config) -> None:
        super().__init__()
        if len(config.hidden_layers) < 2:
            raise ValueError("hidden_layers must contain at least 2 layers for stable residual DNN v3.")
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
                StableResidualBlockV3(
                    trunk_width,
                    dropout=config.dropout,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        self.shared_norm = nn.LayerNorm(trunk_width)
        self.shared_proj = nn.Sequential(
            nn.Linear(trunk_width, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.c_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        self.other_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.output_size - 1),
        )
        self.use_output_sigmoid = config.use_output_sigmoid
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        x = self.blocks(x)
        x = self.shared_norm(x)
        x = self.shared_proj(x)
        c_pred = self.c_head(x)
        other_pred = self.other_head(x)
        output = torch.cat([c_pred, other_pred], dim=1)
        if self.use_output_sigmoid:
            output = self.output_sigmoid(output)
        return output


class SMAResidualDNNV3Trainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAResidualDNNV3Config | None = None) -> None:
        resolved_config = config or SMAResidualDNNV3Config()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_resdnn_v3_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAStableResidualDNNV3Net(self.config).to(self.device)
        self.ema_state: dict[str, torch.Tensor] = {}

    def train_model(self, split_data: dict[str, Any]) -> None:
        train_loader = self.make_loader(split_data["x_train_norm"], split_data["y_train_norm"], shuffle=True)
        criterion = self.make_weighted_mse()
        optimizer = create_torch_optimizer(self.model.parameters(), self.config)
        self.ema_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        total_epochs = max(self.config.epochs, 1)
        warmup_epochs = max(1, min(self.config.warmup_epochs, total_epochs))
        start_epoch = 1

        checkpoint_payload = self.load_training_checkpoint()
        if checkpoint_payload is not None:
            restored = self.restore_training_state(checkpoint_payload, optimizer, scheduler=None)
            start_epoch = restored["start_epoch"]
            best_val_loss = restored["best_val_loss"]
            best_state = restored["best_state"]
            epochs_without_improvement = restored["epochs_without_improvement"]
            extra_state = restored["extra_state"]
            restored_ema = extra_state.get("ema_state")
            if isinstance(restored_ema, dict):
                self.ema_state = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in restored_ema.items()}
            completed_epoch = max(start_epoch - 1, 0)
            print(
                f"Resuming training from epoch {start_epoch} "
                f"(last completed epoch={completed_epoch}, best_val_loss={best_val_loss:.6e}, "
                f"patience_counter={epochs_without_improvement})."
            )

        for epoch in range(start_epoch, total_epochs + 1):
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
                self.update_ema()
                batch_size = features.size(0)
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss = self.compute_loss_with_ema(split_data["x_val_norm"], split_data["y_val_norm"], criterion)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss + self.config.early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in self.ema_state.items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:3d}/{total_epochs:3d} | "
                f"train_loss={train_loss:.6e} | val_loss={val_loss:.6e} | lr={current_lr:.2e}"
            )

            checkpoint_interval = max(int(self.config.checkpoint_every_epochs), 1)
            extra_state = {"ema_state": self.ema_state}
            if epoch % checkpoint_interval == 0 or epoch == total_epochs:
                self.save_training_checkpoint(
                    epoch,
                    optimizer,
                    scheduler=None,
                    best_val_loss=best_val_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    extra_state=extra_state,
                )
            if best_state is not None and abs(best_val_loss - val_loss) <= self.config.early_stopping_min_delta:
                self.save_best_checkpoint(
                    epoch,
                    optimizer,
                    scheduler=None,
                    best_val_loss=best_val_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    extra_state=extra_state,
                )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"after {epoch} epochs (best val_loss={best_val_loss:.6e})."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def update_ema(self) -> None:
        decay = float(self.config.ema_decay)
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                if key not in self.ema_state:
                    self.ema_state[key] = value.detach().clone()
                    continue
                self.ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    def compute_loss_with_ema(self, x: Any, y: Any, criterion: nn.Module) -> float:
        current_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_state)
        try:
            return self.compute_loss(x, y, criterion)
        finally:
            self.model.load_state_dict(current_state)

    def compute_epoch_lr(self, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
        base_lr = self.config.learning_rate
        min_lr = self.config.min_learning_rate
        if epoch <= warmup_epochs:
            return base_lr * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stable deeper residual DNN v3 regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--c-loss-weight", type=float, default=None, help="Override loss weight for coefficient C")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-residual-blocks", type=int, default=None, help="Override number of residual blocks")
    parser.add_argument("--block-width", type=int, default=None, help="Override residual block width")
    args = parser.parse_args()

    config = SMAResidualDNNV3Config()
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

    trainer = SMAResidualDNNV3Trainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
