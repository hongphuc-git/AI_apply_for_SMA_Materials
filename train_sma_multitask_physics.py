from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer


@dataclass
class SMAMultiTaskPhysicsConfig(SMAAnnConfig):
    loading_points: int = 200
    unloading_points: int = 200
    reverse_onset_points: int = 64
    loading_hidden: tuple[int, ...] = (256, 128)
    unloading_hidden: tuple[int, ...] = (256, 128)
    hysteresis_hidden: tuple[int, ...] = (128, 64)
    reverse_onset_hidden: tuple[int, ...] = (128, 64)
    rate_hidden: tuple[int, ...] = (64, 32)
    fusion_hidden: tuple[int, ...] = (256, 128)
    c_head_hidden: tuple[int, ...] = (64, 32)
    l_head_hidden: tuple[int, ...] = (96, 48)
    k_head_hidden: tuple[int, ...] = (96, 48)
    asd_head_hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.06
    learning_rate: float = 4e-4
    weight_decay: float = 1e-5
    c_loss_weight: float = 1.3
    loss_name: str = "huber"
    huber_delta: float = 0.02
    batch_size: int = 64
    epochs: int = 240
    validation_fraction: float = 0.15


def make_mlp(in_features: int, hidden_layers: tuple[int, ...], dropout: float, out_features: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_features
    for width in hidden_layers:
        layers.extend([
            nn.Linear(current, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        current = width
    if out_features is not None:
        layers.append(nn.Linear(current, out_features))
    return nn.Sequential(*layers)


class SMAMultiTaskPhysicsNet(nn.Module):
    def __init__(self, config: SMAMultiTaskPhysicsConfig) -> None:
        super().__init__()
        if config.loading_points + config.unloading_points != config.input_stress_points:
            raise ValueError("loading_points + unloading_points must equal input_stress_points.")
        if config.reverse_onset_points <= 1 or config.reverse_onset_points > config.unloading_points:
            raise ValueError("reverse_onset_points must be between 2 and unloading_points.")

        self.loading_points = config.loading_points
        self.unloading_points = config.unloading_points
        self.reverse_onset_points = config.reverse_onset_points

        self.loading_branch = make_mlp(config.loading_points, config.loading_hidden, config.dropout)
        self.unloading_branch = make_mlp(config.unloading_points, config.unloading_hidden, config.dropout)
        self.hysteresis_branch = make_mlp(config.loading_points, config.hysteresis_hidden, config.dropout)
        self.reverse_onset_branch = make_mlp(config.reverse_onset_points + 6, config.reverse_onset_hidden, config.dropout)
        self.rate_branch = make_mlp(10, config.rate_hidden, config.dropout)

        fusion_in = (
            (config.loading_hidden[-1] if config.loading_hidden else config.loading_points)
            + (config.unloading_hidden[-1] if config.unloading_hidden else config.unloading_points)
            + (config.hysteresis_hidden[-1] if config.hysteresis_hidden else config.loading_points)
            + (config.reverse_onset_hidden[-1] if config.reverse_onset_hidden else (config.reverse_onset_points + 6))
            + (config.rate_hidden[-1] if config.rate_hidden else 10)
        )
        self.fusion_trunk = make_mlp(fusion_in, config.fusion_hidden, config.dropout)
        trunk_out = config.fusion_hidden[-1] if config.fusion_hidden else fusion_in

        self.c_head = make_mlp(trunk_out, config.c_head_hidden, config.dropout, out_features=1)
        self.l_head = make_mlp(trunk_out, config.l_head_hidden, config.dropout, out_features=1)
        self.k_head = make_mlp(trunk_out, config.k_head_hidden, config.dropout, out_features=1)
        self.asd_head = make_mlp(trunk_out, config.asd_head_hidden, config.dropout, out_features=1)
        self.output_sigmoid = nn.Sigmoid() if config.use_output_sigmoid else nn.Identity()

    @staticmethod
    def build_meta_features(loading: torch.Tensor, unloading: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        unloading_reversed = torch.flip(unloading, dims=[1])
        gap = torch.abs(loading - unloading_reversed)
        load_dx = loading[:, 1:] - loading[:, :-1]
        unload_dx = unloading[:, :-1] - unloading[:, 1:]
        peak_stress = torch.max(torch.cat([loading, unloading], dim=1), dim=1, keepdim=True).values
        mean_loading_slope = load_dx.mean(dim=1, keepdim=True)
        mean_unloading_slope = unload_dx.mean(dim=1, keepdim=True)
        loop_area_proxy = gap.mean(dim=1, keepdim=True)
        max_gap = gap.max(dim=1, keepdim=True).values
        mid_idx = loading.size(1) // 2
        mid_gap = torch.abs(loading[:, mid_idx:mid_idx + 1] - unloading_reversed[:, mid_idx:mid_idx + 1])
        late_idx = int(loading.size(1) * 0.8)
        late_gap = torch.abs(loading[:, late_idx:late_idx + 1] - unloading_reversed[:, late_idx:late_idx + 1])
        thermal_rate_proxy = loop_area_proxy * rate
        return torch.cat([
            rate,
            peak_stress,
            mean_loading_slope,
            mean_unloading_slope,
            loop_area_proxy,
            max_gap,
            mid_gap,
            late_gap,
            thermal_rate_proxy,
            peak_stress * rate,
        ], dim=1)

    def build_reverse_onset_features(self, loading: torch.Tensor, unloading: torch.Tensor) -> torch.Tensor:
        unloading_reversed = torch.flip(unloading, dims=[1])
        onset = unloading[:, : self.reverse_onset_points]
        onset_ref = unloading_reversed[:, : self.reverse_onset_points]
        onset_dx = onset[:, :-1] - onset[:, 1:]
        onset_mean_slope = onset_dx.mean(dim=1, keepdim=True)
        onset_max_slope = onset_dx.max(dim=1, keepdim=True).values
        onset_drop = onset[:, :1] - onset[:, -1:]
        onset_gap_mean = torch.abs(loading[:, : self.reverse_onset_points] - onset_ref).mean(dim=1, keepdim=True)
        onset_gap_last = torch.abs(
            loading[:, self.reverse_onset_points - 1:self.reverse_onset_points]
            - onset_ref[:, self.reverse_onset_points - 1:self.reverse_onset_points]
        )
        onset_first = onset[:, :1]
        return torch.cat([onset, onset_mean_slope, onset_max_slope, onset_drop, onset_gap_mean, onset_gap_last, onset_first], dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stress = features[:, :-1]
        rate = features[:, -1:]
        loading = stress[:, : self.loading_points]
        unloading = stress[:, self.loading_points:self.loading_points + self.unloading_points]
        unloading_reversed = torch.flip(unloading, dims=[1])
        hysteresis_signal = loading - unloading_reversed
        reverse_onset_features = self.build_reverse_onset_features(loading, unloading)
        meta_features = self.build_meta_features(loading, unloading, rate)

        loading_repr = self.loading_branch(loading)
        unloading_repr = self.unloading_branch(unloading)
        hysteresis_repr = self.hysteresis_branch(hysteresis_signal)
        reverse_onset_repr = self.reverse_onset_branch(reverse_onset_features)
        rate_repr = self.rate_branch(meta_features)

        fused = torch.cat([loading_repr, unloading_repr, hysteresis_repr, reverse_onset_repr, rate_repr], dim=1)
        trunk = self.fusion_trunk(fused)
        c = self.c_head(trunk)
        l = self.l_head(trunk)
        k = self.k_head(trunk)
        asd = self.asd_head(trunk)
        return self.output_sigmoid(torch.cat([c, l, k, asd], dim=1))


class SMAMultiTaskPhysicsTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAMultiTaskPhysicsConfig | None = None) -> None:
        resolved_config = config or SMAMultiTaskPhysicsConfig()
        super().__init__(root, resolved_config)
        self.config = resolved_config
        self.output_dir = self.root / "python_multitask_physics_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAMultiTaskPhysicsNet(self.config).to(self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physics-guided multi-task SMA regressor.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Workspace root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = SMAMultiTaskPhysicsConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    trainer = SMAMultiTaskPhysicsTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
