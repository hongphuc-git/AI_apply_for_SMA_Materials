from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMATransformerConfig(SMAAnnConfig):
    d_model: int = 96
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    transformer_dropout: float = 0.10
    patch_size: int = 4
    hidden_layers: tuple[int, ...] = (256, 128)
    dropout: float = 0.10
    learning_rate: float = 6e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 3.0
    epochs: int = 180
    pretrained_checkpoint: str | None = None
    freeze_backbone: bool = False


class SMATransformerNet(nn.Module):
    def __init__(self, config: SMATransformerConfig) -> None:
        super().__init__()
        if config.input_stress_points % config.patch_size != 0:
            raise ValueError("input_stress_points must be divisible by patch_size.")

        self.patch_size = config.patch_size
        self.num_patches = config.input_stress_points // config.patch_size
        self.patch_embed = nn.Conv1d(1, config.d_model, kernel_size=config.patch_size, stride=config.patch_size)
        self.rate_token = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)

        head_layers: list[nn.Module] = []
        in_features = config.d_model
        for hidden_size in config.hidden_layers:
            head_layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_features = hidden_size
        head_layers.append(nn.Linear(in_features, config.output_size))
        if config.use_output_sigmoid:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stress_curve = features[:, :-1].unsqueeze(1)
        rate = features[:, -1:].contiguous()
        stress_tokens = self.patch_embed(stress_curve).transpose(1, 2)
        rate_token = self.rate_token(rate).unsqueeze(1)
        tokens = torch.cat([rate_token, stress_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)


class SMATransformerTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMATransformerConfig | None = None) -> None:
        resolved_config = config or SMATransformerConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_transformer_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMATransformerNet(self.config).to(self.device)
        self.maybe_load_checkpoint()

    def maybe_load_checkpoint(self) -> None:
        if not self.config.pretrained_checkpoint:
            return
        checkpoint_path = Path(self.config.pretrained_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.root / checkpoint_path
        payload = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Checkpoint '{checkpoint_path}' does not contain model_state_dict.")
        self.model.load_state_dict(payload["model_state_dict"], strict=False)
        print(f"Loaded checkpoint -> {checkpoint_path}")
        if self.config.freeze_backbone:
            for parameter in self.model.patch_embed.parameters():
                parameter.requires_grad = False
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.model.rate_token.parameters():
                parameter.requires_grad = False
            print("Backbone frozen for fine-tuning.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer regressor for SMA dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--c-loss-weight", type=float, default=None, help="Override loss weight for coefficient C")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="Checkpoint path for fine-tuning")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze transformer backbone when loading a checkpoint")
    args = parser.parse_args()

    config = SMATransformerConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.c_loss_weight is not None:
        config.c_loss_weight = args.c_loss_weight
    if args.pretrained_checkpoint is not None:
        config.pretrained_checkpoint = args.pretrained_checkpoint
    if args.freeze_backbone:
        config.freeze_backbone = True

    trainer = SMATransformerTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
