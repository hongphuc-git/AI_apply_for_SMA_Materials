from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SMAAnnConfig:
    explicit_data_dir: str | None = None
    candidate_data_dirs: tuple[str, ...] = (
        "sma_vector_ann_dataset_test",
        "sma_vector_ann_dataset",
        "sma_ann_stress_vector_dataset",
    )
    input_stress_points: int = 400
    input_size: int = 401
    output_size: int = 4
    hidden_layers: tuple[int, ...] = (512, 256, 128, 64)
    dropout: float = 0.10
    use_output_sigmoid: bool = True
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    optimizer_name: str = "adamw"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_momentum: float = 0.9
    rmsprop_alpha: float = 0.99
    checkpoint_every_epochs: int = 10
    resume_from_dir: str | None = None
    c_loss_weight: float = 2.8
    loss_name: str = "mse"
    huber_delta: float = 0.03
    gradient_clip_norm: float | None = None
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-5
    batch_size: int = 32
    epochs: int = 150
    validation_fraction: float = 0.10
    seed: int = 42
    label_names: tuple[str, ...] = ("C", "L", "k", "Asd")
    target_min: tuple[float, ...] = (800.0, 400.0, 0.001, 247.0)
    target_max: tuple[float, ...] = (2000.0, 40000.0, 0.100, 263.2)


class MinMaxNormalizer:
    def __init__(
        self, y_min: np.ndarray | None = None, y_max: np.ndarray | None = None
    ) -> None:
        self.y_min = None if y_min is None else np.asarray(y_min, dtype=np.float64)
        self.y_max = None if y_max is None else np.asarray(y_max, dtype=np.float64)
        self.y_span: np.ndarray | None = None
        self.zero_span_mask: np.ndarray | None = None
        if self.y_min is not None and self.y_max is not None:
            self._refresh()

    def _refresh(self) -> None:
        if self.y_min is None or self.y_max is None:
            raise ValueError("y_min and y_max must be set before refresh.")
        span = self.y_max - self.y_min
        self.zero_span_mask = span <= 0
        self.y_span = span.copy()
        if self.y_span is None:
            raise ValueError("y_span was not initialized.")
        if self.zero_span_mask is None:
            raise ValueError("zero_span_mask was not initialized.")
        self.y_span[self.zero_span_mask] = 1.0

    def fit(self, values: np.ndarray) -> "MinMaxNormalizer":
        self.y_min = np.min(values, axis=0).astype(np.float64)
        self.y_max = np.max(values, axis=0).astype(np.float64)
        self._refresh()
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.y_min is None or self.y_span is None:
            raise ValueError("Normalizer is not fitted.")
        values_array = np.asarray(values)
        work_dtype = (
            values_array.dtype
            if np.issubdtype(values_array.dtype, np.floating)
            else np.float32
        )
        safe_span = np.asarray(self.y_span, dtype=work_dtype)
        y_min = np.asarray(self.y_min, dtype=work_dtype)
        return (values_array - y_min) / safe_span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.y_min is None or self.y_span is None:
            raise ValueError("Normalizer is not fitted.")
        values_array = np.asarray(values)
        work_dtype = (
            values_array.dtype
            if np.issubdtype(values_array.dtype, np.floating)
            else np.float32
        )
        y_span = np.asarray(self.y_span, dtype=work_dtype)
        y_min = np.asarray(self.y_min, dtype=work_dtype)
        return values_array * y_span + y_min

    def to_dict(self) -> dict[str, list[float]]:
        if (
            self.y_min is None
            or self.y_max is None
            or self.y_span is None
            or self.zero_span_mask is None
        ):
            raise ValueError("Normalizer is not fitted.")
        return {
            "y_min": self.y_min.tolist(),
            "y_max": self.y_max.tolist(),
            "y_span": self.y_span.tolist(),
            "zero_span_mask": self.zero_span_mask.astype(bool).tolist(),
        }


class RegressionMetrics:
    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, label_names: tuple[str, ...]
    ) -> dict[str, Any]:
        mae = np.mean(np.abs(y_pred - y_true), axis=0)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
        r2 = np.full_like(mae, np.nan, dtype=np.float64)
        valid = ss_tot > 0
        r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
        return {
            "label_names": list(label_names),
            "mae": mae.tolist(),
            "rmse": rmse.tolist(),
            "r2": r2.tolist(),
            "mean_mae": float(np.nanmean(mae)),
            "mean_rmse": float(np.nanmean(rmse)),
            "mean_r2": float(np.nanmean(r2)),
        }


def format_rate_levels(rate_values: np.ndarray) -> str:
    unique_rates = np.unique(np.asarray(rate_values, dtype=np.float64))
    if unique_rates.size == 0:
        return "none"
    preview = ", ".join(f"{value:g}" for value in unique_rates[:8])
    if unique_rates.size > 8:
        preview += ", ..."
    return f"{unique_rates.size} levels [{preview}]"


def summarize_split_shapes(split_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_samples": int(split_data["x_train"].shape[0]),
        "val_samples": int(split_data["x_val"].shape[0]),
        "test_samples": int(split_data["x_test"].shape[0]),
        "feature_dim": int(split_data["x_train_norm"].shape[1]),
        "target_dim": int(split_data["y_train_norm"].shape[1]),
        "rate_summary": format_rate_levels(
            np.concatenate(
                [
                    split_data["rate_train"],
                    split_data["rate_val"],
                    split_data["rate_test"],
                ]
            )
        ),
    }


def estimate_runtime_band(
    total_steps: int,
    train_samples: int,
    feature_dim: int,
    model_size_hint: int,
    device_kind: str,
) -> str:
    complexity = max(total_steps, 1) * max(train_samples, 1) * max(feature_dim, 1)
    scaled = complexity * max(model_size_hint, 1)
    if device_kind == "cuda":
        if scaled < 2_500_000_000:
            return "roughly 2-10 minutes on a typical Colab GPU"
        if scaled < 10_000_000_000:
            return "roughly 10-30 minutes on a typical Colab GPU"
        return "roughly 30+ minutes on a typical Colab GPU"
    if scaled < 2_500_000_000:
        return "roughly 10-40 minutes on CPU"
    if scaled < 10_000_000_000:
        return "roughly 40-120 minutes on CPU"
    return "roughly 2+ hours on CPU"


def build_device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        props = torch.cuda.get_device_properties(device_index)
        total_memory_gb = props.total_memory / float(1024**3)
        return f"GPU | {props.name} | VRAM={total_memory_gb:.1f} GB"
    return "CPU | CUDA not available"


def build_hardware_fit_note(
    device: torch.device, batch_size: int, epochs: int, parameter_count: int
) -> str:
    if device.type == "cuda":
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        props = torch.cuda.get_device_properties(device_index)
        total_memory_gb = props.total_memory / float(1024**3)
        if total_memory_gb < 8.0 and batch_size >= 128:
            return "Current batch size may be aggressive for small GPUs; if you hit OOM, try batch_size=32 or 64."
        if total_memory_gb < 12.0 and parameter_count > 3_000_000:
            return "Model is moderately large for this GPU tier; reduce batch_size or use `tabular_resnet`/`mlp_tabular` if memory is tight."
        return "Current configuration looks reasonable for GPU training. If utilization is low, you can try a larger batch size."
    if epochs >= 200 and parameter_count > 1_000_000:
        return "This setup may be slow on CPU; consider `xgboost`, `hist_gradient_boosting`, or lowering epochs/batch size for faster iteration."
    if epochs >= 300:
        return "Long CPU run expected; for quick tests, try fewer epochs or a lighter model such as `mlp_tabular`."
    return "Current configuration should be manageable on CPU, but GPU is recommended for faster training."


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def create_torch_optimizer(
    parameters: Any, config: SMAAnnConfig
) -> torch.optim.Optimizer:
    optimizer_name = str(config.optimizer_name).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.optimizer_betas,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.optimizer_betas,
        )
    if optimizer_name == "nadam":
        return torch.optim.NAdam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.optimizer_betas,
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.optimizer_momentum,
            alpha=config.rmsprop_alpha,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.optimizer_momentum,
            nesterov=config.optimizer_momentum > 0,
        )
    raise ValueError(
        f"Unsupported optimizer_name '{config.optimizer_name}'. Choose from: adamw, adam, nadam, rmsprop, sgd"
    )


class MatDatasetLoader:
    def __init__(self, root: Path, config: SMAAnnConfig) -> None:
        self.root = root
        self.config = config

    def find_data_dir(self) -> Path:
        if self.config.explicit_data_dir:
            explicit_dir = Path(self.config.explicit_data_dir).expanduser()
            if (explicit_dir / "train.mat").is_file() and (
                explicit_dir / "test.mat"
            ).is_file():
                return explicit_dir
            raise FileNotFoundError(
                f"explicit_data_dir does not contain train.mat/test.mat: {explicit_dir}"
            )
        for relative_dir in self.config.candidate_data_dirs:
            data_dir = self.root / relative_dir
            if (data_dir / "train.mat").is_file() and (data_dir / "test.mat").is_file():
                return data_dir
        checked = ", ".join(
            str(self.root / item) for item in self.config.candidate_data_dirs
        )
        raise FileNotFoundError(f"Could not find train.mat/test.mat in: {checked}")

    def load_split(self, file_path: Path) -> dict[str, np.ndarray]:
        try:
            raw = loadmat(file_path)
            return self._from_scipy(raw)
        except NotImplementedError:
            return self._from_hdf5(file_path)

    def _from_scipy(self, raw: dict[str, Any]) -> dict[str, np.ndarray]:
        data = {k: v for k, v in raw.items() if not k.startswith("__")}
        return self._standardize(data)

    def _from_hdf5(self, file_path: Path) -> dict[str, np.ndarray]:
        with h5py.File(file_path, "r") as handle:
            data = {key: np.array(handle[key]) for key in handle.keys()}
        return self._standardize(data)

    def _standardize(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        required = ("X", "Y", "rate")
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")

        x = self._as_2d_samples_first(data["X"], self.config.input_stress_points)
        y = self._as_2d_samples_first(data["Y"], self.config.output_size)
        rate = np.asarray(data["rate"], dtype=np.float64).reshape(-1)
        if x.shape[0] != rate.shape[0] or y.shape[0] != rate.shape[0]:
            raise ValueError(
                f"Mismatched sample counts: X={x.shape[0]}, Y={y.shape[0]}, rate={rate.shape[0]}"
            )

        eps_target = None
        if "epsTarget" in data:
            eps_target = np.asarray(data["epsTarget"], dtype=np.float64).reshape(-1)
            if eps_target.size != self.config.input_stress_points:
                eps_target = None
        if eps_target is None:
            eps_target = np.concatenate(
                [
                    np.linspace(0.0, 0.04, 200, dtype=np.float64),
                    np.linspace(0.04, 0.0, 200, dtype=np.float64),
                ]
            )

        return {
            "X": x.astype(np.float32),
            "Y": y.astype(np.float32),
            "rate": rate.astype(np.float32),
            "epsTarget": eps_target.astype(np.float64),
        }

    @staticmethod
    def _as_2d_samples_first(values: Any, expected_width: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        array = np.squeeze(array)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {array.shape}")
        if array.shape[1] == expected_width:
            return array
        if array.shape[0] == expected_width:
            return array.T
        raise ValueError(
            f"Expected one dimension to equal {expected_width}, got {array.shape}"
        )


class SMAFeedForwardNet(nn.Module):
    def __init__(self, config: SMAAnnConfig) -> None:
        super().__init__()
        if len(config.hidden_layers) < 2:
            raise ValueError("hidden_layers must contain at least 2 layers for DNN.")
        layers: list[nn.Module] = []
        in_features = config.input_size
        for hidden_size in config.hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, config.output_size))
        if config.use_output_sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class WeightedMSELoss(nn.Module):
    def __init__(self, output_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output_weights", output_weights)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared_error = (prediction - target) ** 2
        weights = self.output_weights
        if not isinstance(weights, torch.Tensor):
            raise TypeError("output_weights must be a Tensor.")
        weighted_error = squared_error * weights
        return torch.mean(weighted_error)


class WeightedHuberLoss(nn.Module):
    def __init__(self, output_weights: torch.Tensor, delta: float) -> None:
        super().__init__()
        self.register_buffer("output_weights", output_weights)
        self.delta = float(delta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        abs_error = torch.abs(error)
        quadratic = torch.minimum(
            abs_error,
            torch.tensor(self.delta, device=abs_error.device, dtype=abs_error.dtype),
        )
        linear = abs_error - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear
        weights = self.output_weights
        if not isinstance(weights, torch.Tensor):
            raise TypeError("output_weights must be a Tensor.")
        return torch.mean(huber * weights)


class SMAAnnTrainer:
    def __init__(self, root: Path, config: SMAAnnConfig | None = None) -> None:
        self.root = root
        self.config = config or SMAAnnConfig()
        self.loader = MatDatasetLoader(root, self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = self.loader.find_data_dir()
        self.output_dir = self.root / "python_dnn_outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.model = SMAFeedForwardNet(self.config).to(self.device)
        self.input_normalizer = MinMaxNormalizer()
        self.target_normalizer = MinMaxNormalizer(
            np.asarray(self.config.target_min, dtype=np.float64),
            np.asarray(self.config.target_max, dtype=np.float64),
        )
        self.history = {"train_loss": [], "val_loss": []}
        self.rng = np.random.default_rng(self.config.seed)
        self.run_started_at: float | None = None
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def make_weighted_mse(self) -> nn.Module:
        weights = torch.ones(
            self.config.output_size, dtype=torch.float32, device=self.device
        )
        weights[0] = float(self.config.c_loss_weight)
        if self.config.loss_name == "mse":
            return WeightedMSELoss(weights)
        if self.config.loss_name == "huber":
            return WeightedHuberLoss(weights, self.config.huber_delta)
        raise ValueError(f"Unsupported loss_name '{self.config.loss_name}'.")

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

    def latest_checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint_latest.pt"

    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint_best.pt"

    def print_run_overview(self, split_data: dict[str, Any]) -> None:
        total_params, trainable_params = count_model_parameters(self.model)
        shape_summary = summarize_split_shapes(split_data)
        steps_per_epoch = max(
            1,
            math.ceil(
                shape_summary["train_samples"] / max(int(self.config.batch_size), 1)
            ),
        )
        total_steps = steps_per_epoch * max(int(self.config.epochs), 1)
        print("Run overview")
        print(f"  Data directory      : {self.data_dir}")
        print("  Data files         : train.mat + test.mat")
        print(
            f"  Input format       : X + rate -> {shape_summary['feature_dim']} features per sample"
        )
        print(
            f"  Target format      : Y -> {shape_summary['target_dim']} regression targets {self.config.label_names}"
        )
        print(
            f"  Dataset split      : train={shape_summary['train_samples']}, val={shape_summary['val_samples']}, test={shape_summary['test_samples']}"
        )
        print(f"  Rate coverage      : {shape_summary['rate_summary']}")
        print(f"  Device             : {build_device_summary(self.device)}")
        print(f"  Model              : {self.model.__class__.__name__}")
        print(
            f"  Parameters         : total={total_params:,}, trainable={trainable_params:,}"
        )
        print(
            f"  Config             : epochs={self.config.epochs}, batch_size={self.config.batch_size}, lr={self.config.learning_rate:g}, optimizer={self.config.optimizer_name}"
        )
        print(
            f"  Steps              : {steps_per_epoch} per epoch, {total_steps} total"
        )
        print(
            "  Time estimate      : "
            + estimate_runtime_band(
                total_steps,
                shape_summary["train_samples"],
                shape_summary["feature_dim"],
                total_params,
                self.device.type,
            )
        )
        print(
            "  Hardware fit       : "
            + build_hardware_fit_note(
                self.device,
                int(self.config.batch_size),
                int(self.config.epochs),
                total_params,
            )
        )
        resume_checkpoint = self.find_resume_checkpoint_path()
        if resume_checkpoint is not None:
            print(f"  Resume mode        : auto-resume from {resume_checkpoint}")
        print("  Architecture")
        print(self.model)

    def find_resume_checkpoint_path(self) -> Path | None:
        checkpoint_path = self.latest_checkpoint_path()
        if checkpoint_path.is_file():
            return checkpoint_path
        if self.config.resume_from_dir:
            print(f"Resume requested but checkpoint not found: {checkpoint_path}")
        return None

    def build_checkpoint_payload(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        best_val_loss: float,
        best_state: dict[str, torch.Tensor] | None,
        epochs_without_improvement: int,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None
            if scheduler is None
            else scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_state_dict": best_state,
            "epochs_without_improvement": epochs_without_improvement,
            "history": self.history,
            "config": asdict(self.config),
            "input_normalizer": self.input_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
            "extra_state": extra_state or {},
        }

    def save_training_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        best_val_loss: float,
        best_state: dict[str, torch.Tensor] | None,
        epochs_without_improvement: int,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        torch.save(
            self.build_checkpoint_payload(
                epoch,
                optimizer,
                scheduler,
                best_val_loss,
                best_state,
                epochs_without_improvement,
                extra_state,
            ),
            self.latest_checkpoint_path(),
        )

    def save_best_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        best_val_loss: float,
        best_state: dict[str, torch.Tensor],
        epochs_without_improvement: int,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        payload = self.build_checkpoint_payload(
            epoch,
            optimizer,
            scheduler,
            best_val_loss,
            best_state,
            epochs_without_improvement,
            extra_state,
        )
        payload["model_state_dict"] = best_state
        torch.save(payload, self.best_checkpoint_path())

    def load_training_checkpoint(self) -> dict[str, Any] | None:
        checkpoint_path = self.find_resume_checkpoint_path()
        if checkpoint_path is None:
            return None
        return torch.load(checkpoint_path, map_location=self.device)

    def restore_training_state(
        self,
        checkpoint_payload: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> dict[str, Any]:
        self.model.load_state_dict(checkpoint_payload["model_state_dict"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, self.device)
        scheduler_state_dict = checkpoint_payload.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
        self.history = checkpoint_payload.get(
            "history", {"train_loss": [], "val_loss": []}
        )
        return {
            "start_epoch": int(checkpoint_payload.get("epoch", 0)) + 1,
            "best_val_loss": float(
                checkpoint_payload.get("best_val_loss", float("inf"))
            ),
            "best_state": checkpoint_payload.get("best_state_dict"),
            "epochs_without_improvement": int(
                checkpoint_payload.get("epochs_without_improvement", 0)
            ),
            "extra_state": checkpoint_payload.get("extra_state", {}),
        }

    def prepare_datasets(
        self, train_raw: dict[str, np.ndarray], test_raw: dict[str, np.ndarray]
    ) -> dict[str, Any]:
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

    def stratified_split(
        self, rate: np.ndarray, train_target: int, val_target: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
                raise ValueError(
                    f"Not enough samples for rate {level:.4f}. Need {n_needed}, found {idx_rate.size}."
                )
            train_indices.extend(idx_rate[:n_train_rate].tolist())
            val_indices.extend(idx_rate[n_train_rate:n_needed].tolist())
        train_indices_arr = self.rng.permutation(
            np.asarray(train_indices, dtype=np.int64)
        )
        val_indices_arr = self.rng.permutation(np.asarray(val_indices, dtype=np.int64))
        return train_indices_arr, val_indices_arr

    @staticmethod
    def distribute_counts(total_count: int, n_bins: int) -> np.ndarray:
        base_count = total_count // n_bins
        counts = np.full(n_bins, base_count, dtype=np.int64)
        remainder = total_count - base_count * n_bins
        if remainder > 0:
            counts[:remainder] += 1
        return counts

    def train_model(self, split_data: dict[str, Any]) -> None:
        train_loader = self.make_loader(
            split_data["x_train_norm"], split_data["y_train_norm"], shuffle=True
        )
        criterion = self.make_weighted_mse()
        optimizer = create_torch_optimizer(self.model.parameters(), self.config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        start_epoch = 1
        checkpoint_payload = self.load_training_checkpoint()
        if checkpoint_payload is not None:
            restored = self.restore_training_state(
                checkpoint_payload, optimizer, scheduler
            )
            start_epoch = restored["start_epoch"]
            best_val_loss = restored["best_val_loss"]
            best_state = restored["best_state"]
            epochs_without_improvement = restored["epochs_without_improvement"]
            completed_epoch = max(start_epoch - 1, 0)
            print(
                f"Resuming training from epoch {start_epoch} "
                f"(last completed epoch={completed_epoch}, best_val_loss={best_val_loss:.6e}, "
                f"patience_counter={epochs_without_improvement})."
            )
        for epoch in range(start_epoch, self.config.epochs + 1):
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
                if (
                    self.config.gradient_clip_norm is not None
                    and self.config.gradient_clip_norm > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                optimizer.step()
                batch_size = features.size(0)
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size
            train_loss = train_loss_sum / max(train_count, 1)
            val_loss = self.compute_loss(
                split_data["x_val_norm"], split_data["y_val_norm"], criterion
            )
            scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss + self.config.early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{self.config.epochs:3d} | "
                f"train_loss={train_loss:.6e} | val_loss={val_loss:.6e} | lr={current_lr:.2e}"
            )

            checkpoint_interval = max(int(self.config.checkpoint_every_epochs), 1)
            if epoch % checkpoint_interval == 0 or epoch == self.config.epochs:
                self.save_training_checkpoint(
                    epoch,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    best_state,
                    epochs_without_improvement,
                )
            if (
                best_state is not None
                and abs(best_val_loss - val_loss)
                <= self.config.early_stopping_min_delta
            ):
                self.save_best_checkpoint(
                    epoch,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    best_state,
                    epochs_without_improvement,
                )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"after {epoch} epochs (best val_loss={best_val_loss:.6e})."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def make_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        import platform

        dataset = TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        # Windows multiprocessing with DataLoader requires the spawn start method which
        # adds overhead; default to 0 workers there to avoid it. On Linux/macOS (Colab
        # included) use 2 workers so data prefetch overlaps with GPU compute.
        use_workers = 0 if platform.system() == "Windows" else 2
        use_pin_memory = self.device.type == "cuda"
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=use_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(use_workers > 0),
            prefetch_factor=2 if use_workers > 0 else None,
        )

    def compute_loss(self, x: np.ndarray, y: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        loader = self.make_loader(x, y, shuffle=False)
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(features)
                total_loss += float(
                    criterion(predictions, targets).item()
                ) * features.size(0)
                total_count += features.size(0)
        return total_loss / max(total_count, 1)

    @staticmethod
    def constrain_normalized_predictions(predictions: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(predictions, dtype=np.float32), 0.0, 1.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        outputs: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for (features,) in loader:
                features = features.to(self.device)
                outputs.append(self.model(features).cpu().numpy())
        return self.constrain_normalized_predictions(np.concatenate(outputs, axis=0))

    def evaluate(self, split_data: dict[str, Any]) -> dict[str, Any]:
        y_train_pred_norm = self.predict(split_data["x_train_norm"])
        y_val_pred_norm = self.predict(split_data["x_val_norm"])
        y_test_pred_norm = self.predict(split_data["x_test_norm"])
        y_train_pred = self.target_normalizer.inverse_transform(y_train_pred_norm)
        y_val_pred = self.target_normalizer.inverse_transform(y_val_pred_norm)
        y_test_pred = self.target_normalizer.inverse_transform(y_test_pred_norm)
        return {
            "split_data": split_data,
            "history": self.history,
            "train_metrics_norm": RegressionMetrics.evaluate(
                split_data["y_train_norm"], y_train_pred_norm, self.config.label_names
            ),
            "val_metrics_norm": RegressionMetrics.evaluate(
                split_data["y_val_norm"], y_val_pred_norm, self.config.label_names
            ),
            "test_metrics_norm": RegressionMetrics.evaluate(
                split_data["y_test_norm"], y_test_pred_norm, self.config.label_names
            ),
            "train_metrics": RegressionMetrics.evaluate(
                split_data["y_train"], y_train_pred, self.config.label_names
            ),
            "val_metrics": RegressionMetrics.evaluate(
                split_data["y_val"], y_val_pred, self.config.label_names
            ),
            "test_metrics": RegressionMetrics.evaluate(
                split_data["y_test"], y_test_pred, self.config.label_names
            ),
            "y_train_pred_norm": y_train_pred_norm,
            "y_val_pred_norm": y_val_pred_norm,
            "y_test_pred_norm": y_test_pred_norm,
            "y_train_pred": y_train_pred,
            "y_val_pred": y_val_pred,
            "y_test_pred": y_test_pred,
        }

    def save_artifacts(self, artifacts: dict[str, Any]) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "input_normalizer": self.input_normalizer.to_dict(),
                "target_normalizer": self.target_normalizer.to_dict(),
            },
            self.output_dir / "sma_dnn_regressor.pt",
        )
        metrics_payload = {
            "config": asdict(self.config),
            "history": self.history,
            "input_normalizer": self.input_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
            "train_metrics_norm": artifacts["train_metrics_norm"],
            "val_metrics_norm": artifacts["val_metrics_norm"],
            "test_metrics_norm": artifacts["test_metrics_norm"],
            "train_metrics": artifacts["train_metrics"],
            "val_metrics": artifacts["val_metrics"],
            "test_metrics": artifacts["test_metrics"],
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="ascii"
        )
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
        self._save_loss_plot()
        self._save_scatter_plot(
            artifacts["split_data"]["y_test"],
            artifacts["y_test_pred"],
            artifacts["test_metrics"]["r2"],
            "physical",
            False,
        )
        self._save_scatter_plot(
            artifacts["split_data"]["y_test_norm"],
            artifacts["y_test_pred_norm"],
            artifacts["test_metrics_norm"]["r2"],
            "normalized",
            True,
        )

    def _save_loss_plot(self) -> None:
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            epochs,
            self.history["train_loss"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="Train",
        )
        ax.plot(
            epochs,
            self.history["val_loss"],
            marker="s",
            linewidth=1.5,
            markersize=3,
            label="Validation",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training and Validation Loss")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)

    def _save_scatter_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        r2_values: list[float],
        suffix: str,
        normalized: bool,
    ) -> None:
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
            axis.set_title(
                f"{self.config.label_names[idx]} | R^2 = {r2_values[idx]:.4f}"
            )
            axis.grid(True)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"test_scatter_{suffix}.png", dpi=160)
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
        print("\nExample test predictions:")
        n_examples = min(5, artifacts["split_data"]["x_test"].shape[0])
        for idx in range(n_examples):
            rate = artifacts["split_data"]["rate_test"][idx]
            true_values = artifacts["split_data"]["y_test"][idx]
            pred_values = artifacts["y_test_pred"][idx]
            print(f"Sample {idx + 1} | rate = {rate:.2f} %/s")
            print(
                f"  true = [{true_values[0]:10.4f} {true_values[1]:10.4f} {true_values[2]:10.6f} {true_values[3]:10.4f}]"
            )
            print(
                f"  pred = [{pred_values[0]:10.4f} {pred_values[1]:10.4f} {pred_values[2]:10.6f} {pred_values[3]:10.4f}]"
            )
        print(f"\nSaved Python model artifacts -> {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DNN regressor for SMA dataset with C-focused weighted loss."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--c-loss-weight",
        type=float,
        default=None,
        help="Override loss weight for coefficient C",
    )
    args = parser.parse_args()
    config = SMAAnnConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.c_loss_weight is not None:
        config.c_loss_weight = args.c_loss_weight
    trainer = SMAAnnTrainer(args.root.resolve(), config)
    trainer.run()


if __name__ == "__main__":
    main()
