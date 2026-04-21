from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from train_sma_ann import SMAAnnConfig, SMAAnnTrainer
from train_sma_cnn import SMACNNConfig, SMACNNTrainer
from train_sma_c_physics import SMACPhysicsConfig, SMACPhysicsTrainer
from train_sma_multibranch import SMAMultiBranchConfig, SMAMultiBranchTrainer
from train_sma_multitask_physics import SMAMultiTaskPhysicsConfig, SMAMultiTaskPhysicsTrainer
from train_sma_resdnn import SMAResidualDNNConfig, SMAResidualDNNTrainer
from train_sma_resdnn_v2 import SMAResidualDNNV2Config, SMAResidualDNNV2Trainer
from train_sma_resdnn_v3 import SMAResidualDNNV3Config, SMAResidualDNNV3Trainer
from train_sma_sklearn_ensemble import (
    SMASklearnEnsembleConfig,
    SMASklearnEnsembleTrainer,
)
from train_sma_transformer import SMATransformerConfig, SMATransformerTrainer
from train_sma_xgboost import SMAXGBoostConfig, SMAXGBoostTrainer


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", tag.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "run"


def build_run_name(
    model_name: str, timestamp: datetime | None = None, tag: str | None = None
) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{sanitize_tag(tag)}" if tag else ""
    return f"{model_name}_{stamp}{suffix}"


def parse_override_value(current_value: Any, new_value: Any) -> Any:
    if isinstance(current_value, tuple) and isinstance(new_value, list):
        return tuple(new_value)
    if isinstance(current_value, bool) and isinstance(new_value, str):
        lowered = new_value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return new_value


def apply_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    unknown_keys = [key for key in overrides if not hasattr(config, key)]
    if unknown_keys:
        raise KeyError(f"Unknown config override keys: {unknown_keys}")
    for key, value in overrides.items():
        current_value = getattr(config, key)
        setattr(config, key, parse_override_value(current_value, value))
    return config


def load_overrides(config_json: str | None, config_file: Path | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if config_file is not None:
        overrides.update(json.loads(config_file.read_text(encoding="utf-8")))
    if config_json:
        overrides.update(json.loads(config_json))
    return overrides


class TimestampedANNTrainer(SMAAnnTrainer):
    def __init__(self, root: Path, config: SMAAnnConfig, output_dir: Path) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedCNNTrainer(SMACNNTrainer):
    def __init__(self, root: Path, config: SMACNNConfig, output_dir: Path) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedCPhysicsTrainer(SMACPhysicsTrainer):
    def __init__(self, root: Path, config: SMACPhysicsConfig, output_dir: Path) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedMultiBranchTrainer(SMAMultiBranchTrainer):
    def __init__(self, root: Path, config: SMAMultiBranchConfig, output_dir: Path) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedMultiTaskPhysicsTrainer(SMAMultiTaskPhysicsTrainer):
    def __init__(self, root: Path, config: SMAMultiTaskPhysicsConfig, output_dir: Path) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedResidualDNNTrainer(SMAResidualDNNTrainer):
    def __init__(
        self, root: Path, config: SMAResidualDNNConfig, output_dir: Path
    ) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedResidualDNNV2Trainer(SMAResidualDNNV2Trainer):
    def __init__(
        self, root: Path, config: SMAResidualDNNV2Config, output_dir: Path
    ) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedResidualDNNV3Trainer(SMAResidualDNNV3Trainer):
    def __init__(
        self, root: Path, config: SMAResidualDNNV3Config, output_dir: Path
    ) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedSklearnEnsembleTrainer(SMASklearnEnsembleTrainer):
    def __init__(
        self, root: Path, config: SMASklearnEnsembleConfig, output_dir: Path
    ) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TimestampedTransformerTrainer(SMATransformerTrainer):
    def __init__(
        self, root: Path, config: SMATransformerConfig, output_dir: Path
    ) -> None:
        super().__init__(root, config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


TrainerFactory = Callable[[Path, Any, Path], Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_cls: type[Any]
    trainer_cls: type[Any]
    description: str
    task_family: str
    supports_train_optimizer: bool = True
    base_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchStrategySpec:
    name: str
    description: str
    sampler_factory_name: str | None
    requires_search_space: bool = True


MODEL_SPEC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ann": {
        "name": "ann",
        "config_cls": SMAAnnConfig,
        "trainer_cls": TimestampedANNTrainer,
        "description": "Baseline weighted DNN/MLP trainer",
        "task_family": "neural",
    },
    "mlp_tabular": {
        "name": "mlp_tabular",
        "config_cls": SMAAnnConfig,
        "trainer_cls": TimestampedANNTrainer,
        "description": "Wider MLP tuned for tabular-style SMA features",
        "task_family": "neural",
        "base_overrides": {
            "hidden_layers": (768, 384, 192, 96),
            "dropout": 0.08,
            "learning_rate": 6e-4,
            "weight_decay": 2e-5,
            "batch_size": 64,
            "epochs": 180,
        },
    },
    "cnn": {
        "name": "cnn",
        "config_cls": SMACNNConfig,
        "trainer_cls": TimestampedCNNTrainer,
        "description": "1D CNN over stress curve plus rate feature",
        "task_family": "neural",
    },
    "c_physics": {
        "name": "c_physics",
        "config_cls": SMACPhysicsConfig,
        "trainer_cls": TimestampedCPhysicsTrainer,
        "description": "C-only physics-guided network using loading, hysteresis, and thermal proxy branches",
        "task_family": "neural",
        "base_overrides": {
            "loading_hidden": (256, 128),
            "hysteresis_hidden": (128, 64),
            "rate_hidden": (48, 24),
            "thermal_hidden": (96, 48),
            "fusion_hidden": (128, 64),
            "dropout": 0.05,
            "learning_rate": 4e-4,
            "batch_size": 64,
            "epochs": 220
        },
    },
    "multitask_physics": {
        "name": "multitask_physics",
        "config_cls": SMAMultiTaskPhysicsConfig,
        "trainer_cls": TimestampedMultiTaskPhysicsTrainer,
        "description": "Physics-guided multi-head network for C, L, k, Asd with specialized branches and heads",
        "task_family": "neural",
        "base_overrides": {
            "loading_hidden": (256, 128),
            "unloading_hidden": (256, 128),
            "hysteresis_hidden": (128, 64),
            "reverse_onset_hidden": (128, 64),
            "rate_hidden": (64, 32),
            "fusion_hidden": (256, 128),
            "c_head_hidden": (64, 32),
            "l_head_hidden": (96, 48),
            "k_head_hidden": (96, 48),
            "asd_head_hidden": (64, 32),
            "dropout": 0.06,
            "learning_rate": 4e-4,
            "batch_size": 64,
            "epochs": 240,
            "c_loss_weight": 1.3
        },
    },
    "multibranch": {
        "name": "multibranch",
        "config_cls": SMAMultiBranchConfig,
        "trainer_cls": TimestampedMultiBranchTrainer,
        "description": "Multi-branch network with separate loading, unloading, and rate encoders",
        "task_family": "neural",
        "base_overrides": {
            "loading_hidden": (256, 128),
            "unloading_hidden": (256, 128),
            "rate_hidden": (32, 16),
            "fusion_hidden": (256, 128),
            "dropout": 0.06,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "epochs": 220,
            "c_loss_weight": 2.5,
        },
    },
    "resdnn": {
        "name": "resdnn",
        "config_cls": SMAResidualDNNConfig,
        "trainer_cls": TimestampedResidualDNNTrainer,
        "description": "Residual DNN baseline with MLP residual blocks",
        "task_family": "neural",
    },
    "tabular_resnet": {
        "name": "tabular_resnet",
        "config_cls": SMAResidualDNNConfig,
        "trainer_cls": TimestampedResidualDNNTrainer,
        "description": "Residual tabular network tuned for structured SMA features",
        "task_family": "neural",
        "base_overrides": {
            "hidden_layers": (640, 320),
            "block_width": 320,
            "num_residual_blocks": 5,
            "dropout": 0.04,
            "learning_rate": 2e-4,
            "batch_size": 96,
            "epochs": 260,
        },
    },
    "resdnn_v2": {
        "name": "resdnn_v2",
        "config_cls": SMAResidualDNNV2Config,
        "trainer_cls": TimestampedResidualDNNV2Trainer,
        "description": "Deeper stable residual DNN with pre-norm blocks and warmup-cosine schedule",
        "task_family": "neural",
    },
    "resdnn_v3": {
        "name": "resdnn_v3",
        "config_cls": SMAResidualDNNV3Config,
        "trainer_cls": TimestampedResidualDNNV3Trainer,
        "description": "Stable residual DNN v3 with C-specialized head and EMA-smoothed validation",
        "task_family": "neural",
    },
    "xgboost": {
        "name": "xgboost",
        "config_cls": SMAXGBoostConfig,
        "trainer_cls": SMAXGBoostTrainer,
        "description": "Per-target XGBoost baseline",
        "task_family": "tree",
        "supports_train_optimizer": False,
    },
    "transformer": {
        "name": "transformer",
        "config_cls": SMATransformerConfig,
        "trainer_cls": TimestampedTransformerTrainer,
        "description": "Patch-based Transformer over stress sequence with optional checkpoint fine-tuning",
        "task_family": "neural",
    },
    "random_forest": {
        "name": "random_forest",
        "config_cls": SMASklearnEnsembleConfig,
        "trainer_cls": TimestampedSklearnEnsembleTrainer,
        "description": "Random forest multi-output baseline",
        "task_family": "tree",
        "supports_train_optimizer": False,
        "base_overrides": {
            "estimator_name": "random_forest",
            "output_dir_name": "python_random_forest_outputs",
        },
    },
    "extra_trees": {
        "name": "extra_trees",
        "config_cls": SMASklearnEnsembleConfig,
        "trainer_cls": TimestampedSklearnEnsembleTrainer,
        "description": "Extra trees multi-output baseline",
        "task_family": "tree",
        "supports_train_optimizer": False,
        "base_overrides": {
            "estimator_name": "extra_trees",
            "output_dir_name": "python_extra_trees_outputs",
        },
    },
    "gradient_boosting": {
        "name": "gradient_boosting",
        "config_cls": SMASklearnEnsembleConfig,
        "trainer_cls": TimestampedSklearnEnsembleTrainer,
        "description": "Gradient boosting tabular baseline wrapped for multi-output regression",
        "task_family": "tree",
        "supports_train_optimizer": False,
        "base_overrides": {
            "estimator_name": "gradient_boosting",
            "output_dir_name": "python_gradient_boosting_outputs",
            "n_estimators": 400,
            "learning_rate": 0.04,
            "max_depth": 3,
            "subsample": 0.9,
        },
    },
    "hist_gradient_boosting": {
        "name": "hist_gradient_boosting",
        "config_cls": SMASklearnEnsembleConfig,
        "trainer_cls": TimestampedSklearnEnsembleTrainer,
        "description": "Histogram gradient boosting baseline for larger tabular datasets",
        "task_family": "tree",
        "supports_train_optimizer": False,
        "base_overrides": {
            "estimator_name": "hist_gradient_boosting",
            "output_dir_name": "python_hist_gradient_boosting_outputs",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 8,
            "max_leaf_nodes": 31,
        },
    },
    "ada_boost": {
        "name": "ada_boost",
        "config_cls": SMASklearnEnsembleConfig,
        "trainer_cls": TimestampedSklearnEnsembleTrainer,
        "description": "AdaBoost baseline for lightweight tabular comparison runs",
        "task_family": "tree",
        "supports_train_optimizer": False,
        "base_overrides": {
            "estimator_name": "ada_boost",
            "output_dir_name": "python_ada_boost_outputs",
            "n_estimators": 300,
            "learning_rate": 0.05,
        },
    },
}

MODEL_REGISTRY: dict[str, ModelSpec] = {
    name: ModelSpec(**spec) for name, spec in MODEL_SPEC_DEFINITIONS.items()
}

SEARCH_STRATEGY_REGISTRY: dict[str, SearchStrategySpec] = {
    "none": SearchStrategySpec(
        name="none",
        description="Run a single experiment without hyperparameter search",
        sampler_factory_name=None,
        requires_search_space=False,
    ),
    "optuna-tpe": SearchStrategySpec(
        name="optuna-tpe",
        description="Optuna TPE sampler for Bayesian-style search",
        sampler_factory_name="TPESampler",
    ),
    "optuna-random": SearchStrategySpec(
        name="optuna-random",
        description="Optuna random sampler for baseline search",
        sampler_factory_name="RandomSampler",
    ),
}


DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "ann": {
        "learning_rate": {"type": "float", "low": 3e-4, "high": 2e-3, "log": True},
        "dropout": {"type": "float", "low": 0.03, "high": 0.20},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.5},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "mlp_tabular": {
        "learning_rate": {"type": "float", "low": 2e-4, "high": 1.2e-3, "log": True},
        "dropout": {"type": "float", "low": 0.02, "high": 0.15},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 5e-4, "log": True},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.5},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "cnn": {
        "learning_rate": {"type": "float", "low": 2e-4, "high": 2e-3, "log": True},
        "cnn_dropout": {"type": "float", "low": 0.03, "high": 0.20},
        "dropout": {"type": "float", "low": 0.03, "high": 0.20},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.5},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "resdnn": {
        "learning_rate": {"type": "float", "low": 2e-4, "high": 2e-3, "log": True},
        "dropout": {"type": "float", "low": 0.03, "high": 0.18},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.5},
        "num_residual_blocks": {"type": "int", "low": 2, "high": 6},
        "block_width": {"type": "categorical", "choices": [128, 192, 256, 384]},
    },
    "tabular_resnet": {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 8e-4, "log": True},
        "dropout": {"type": "float", "low": 0.02, "high": 0.12},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.0},
        "num_residual_blocks": {"type": "int", "low": 3, "high": 7},
        "block_width": {"type": "categorical", "choices": [192, 256, 320, 384]},
        "batch_size": {"type": "categorical", "choices": [64, 96, 128]},
    },
    "resdnn_v2": {
        "learning_rate": {"type": "float", "low": 8e-5, "high": 4e-4, "log": True},
        "dropout": {"type": "float", "low": 0.01, "high": 0.08},
        "c_loss_weight": {"type": "float", "low": 1.8, "high": 3.5},
        "num_residual_blocks": {"type": "int", "low": 4, "high": 8},
        "block_width": {"type": "categorical", "choices": [256, 320, 384, 448]},
        "batch_size": {"type": "categorical", "choices": [64, 96, 128]},
    },
    "resdnn_v3": {
        "learning_rate": {"type": "float", "low": 6e-5, "high": 5e-4, "log": True},
        "dropout": {"type": "float", "low": 0.01, "high": 0.08},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 5.0},
        "num_residual_blocks": {"type": "int", "low": 4, "high": 10},
        "block_width": {"type": "categorical", "choices": [256, 320, 384, 448, 512]},
        "batch_size": {"type": "categorical", "choices": [64, 96, 128, 160]},
        "ema_decay": {"type": "float", "low": 0.99, "high": 0.999},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    },
    "xgboost": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 10},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 8.0},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
    },
    "transformer": {
        "learning_rate": {"type": "float", "low": 2e-4, "high": 2e-3, "log": True},
        "transformer_dropout": {"type": "float", "low": 0.03, "high": 0.18},
        "dropout": {"type": "float", "low": 0.03, "high": 0.18},
        "c_loss_weight": {"type": "float", "low": 1.5, "high": 4.5},
        "d_model": {"type": "categorical", "choices": [64, 96, 128]},
        "nhead": {"type": "categorical", "choices": [4, 8]},
        "num_layers": {"type": "int", "low": 2, "high": 6},
        "dim_feedforward": {"type": "categorical", "choices": [128, 256, 384, 512]},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 200, "high": 800},
        "max_depth": {"type": "categorical", "choices": [None, 12, 18, 24, 32]},
        "min_samples_split": {"type": "int", "low": 2, "high": 12},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 6},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.5, 0.8, 1.0],
        },
    },
    "extra_trees": {
        "n_estimators": {"type": "int", "low": 200, "high": 800},
        "max_depth": {"type": "categorical", "choices": [None, 12, 18, 24, 32]},
        "min_samples_split": {"type": "int", "low": 2, "high": 12},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 6},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.5, 0.8, 1.0],
        },
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "low": 150, "high": 600},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.12, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 6},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 6},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "hist_gradient_boosting": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.12, "log": True},
        "max_depth": {"type": "categorical", "choices": [None, 4, 6, 8, 12]},
        "min_samples_leaf": {"type": "int", "low": 10, "high": 40},
        "max_leaf_nodes": {"type": "categorical", "choices": [15, 31, 63, 127]},
        "l2_regularization": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
    },
    "ada_boost": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    },
}


def create_run_dir(
    root: Path, runs_root: str, model_name: str, tag: str | None
) -> Path:
    run_dir = root / runs_root / build_run_name(model_name, tag=tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def path_relative_to_root_if_possible(path: Path, root: Path) -> Path:
    resolved_root = root.expanduser().resolve(strict=False)
    resolved_path = path.expanduser().resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError:
        return resolved_path


def write_run_manifest(
    run_dir: Path, model_name: str, config: Any, overrides: dict[str, Any]
) -> None:
    payload = {
        "model": model_name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "overrides": overrides,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="ascii"
    )


def get_model_config(model_name: str, overrides: dict[str, Any] | None = None) -> Any:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unsupported model '{model_name}'. Choose from: {sorted(MODEL_REGISTRY)}"
        )
    model_spec = MODEL_REGISTRY[model_name]
    config = model_spec.config_cls()
    merged_overrides: dict[str, Any] = dict(model_spec.base_overrides)
    if overrides:
        merged_overrides.update(overrides)
    apply_overrides(config, merged_overrides)
    return config


def read_metrics_value(metrics_file: Path, metric_path: str) -> float:
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    current: Any = payload
    for key in metric_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Metric path '{metric_path}' not found in {metrics_file}")
        current = current[key]
    return float(current)


def suggest_value(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    spec_type = spec["type"]
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if spec_type == "int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(
        f"Unsupported search space type '{spec_type}' for parameter '{name}'."
    )


def sample_search_overrides(
    trial: Any, search_space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        name: suggest_value(trial, name, spec) for name, spec in search_space.items()
    }


def load_search_space(
    model_name: str, search_space_json: str | None, search_space_file: Path | None
) -> dict[str, dict[str, Any]]:
    search_space = dict(DEFAULT_SEARCH_SPACES.get(model_name, {}))
    if search_space_file is not None:
        search_space.update(json.loads(search_space_file.read_text(encoding="utf-8")))
    if search_space_json:
        search_space.update(json.loads(search_space_json))
    if not search_space:
        raise ValueError(f"No search space is defined for model '{model_name}'.")
    return search_space


def get_search_strategy(strategy_name: str) -> SearchStrategySpec:
    if strategy_name not in SEARCH_STRATEGY_REGISTRY:
        raise KeyError(
            f"Unsupported search strategy '{strategy_name}'. Choose from: {sorted(SEARCH_STRATEGY_REGISTRY)}"
        )
    return SEARCH_STRATEGY_REGISTRY[strategy_name]


def create_optuna_sampler(optuna_module: Any, strategy: SearchStrategySpec) -> Any:
    if strategy.sampler_factory_name is None:
        return None
    sampler_cls = getattr(optuna_module.samplers, strategy.sampler_factory_name)
    if strategy.sampler_factory_name == "TPESampler":
        return sampler_cls(seed=42, multivariate=True, n_startup_trials=5)
    return sampler_cls(seed=42)


def write_trials_csv(output_file: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_run_summaries(search_root: Path, metric_path: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for metrics_file in sorted(search_root.glob("**/metrics.json")):
        run_dir = metrics_file.parent
        manifest_file = run_dir / "run_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        try:
            score = read_metrics_value(metrics_file, metric_path)
        except Exception:
            continue
        summaries.append(
            {
                "model": manifest.get("model", run_dir.name),
                "run_dir": str(run_dir),
                "created_at": manifest.get("created_at", ""),
                "score": score,
            }
        )
    return summaries


def write_leaderboard(search_root: Path, metric_path: str, direction: str) -> Path:
    rows = collect_run_summaries(search_root, metric_path)
    reverse = direction == "maximize"
    rows.sort(key=lambda item: item["score"], reverse=reverse)
    leaderboard_file = search_root / "leaderboard.csv"
    write_trials_csv(leaderboard_file, rows)
    (search_root / "leaderboard.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return leaderboard_file


def run_experiment(
    model_name: str,
    root: Path,
    runs_root: str = "colab_runs",
    tag: str | None = None,
    overrides: dict[str, Any] | None = None,
    optuna_trial: Any = None,
) -> Path:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unsupported model '{model_name}'. Choose from: {sorted(MODEL_REGISTRY)}"
        )

    model_spec = MODEL_REGISTRY[model_name]
    merged_overrides: dict[str, Any] = dict(model_spec.base_overrides)
    if overrides:
        merged_overrides.update(overrides)
    config = get_model_config(model_name, merged_overrides)

    resume_from_dir = merged_overrides.get("resume_from_dir")
    if resume_from_dir and model_spec.task_family != "neural":
        raise ValueError(
            "Checkpoint resume is currently supported for neural/PyTorch models only."
        )

    if isinstance(resume_from_dir, str) and resume_from_dir.strip():
        run_dir = Path(resume_from_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(root, runs_root, model_name, tag)
    write_run_manifest(run_dir, model_name, config, merged_overrides)

    trainer_cls = model_spec.trainer_cls
    if model_name == "xgboost":
        config.output_dir_name = str(path_relative_to_root_if_possible(run_dir, root))
        trainer = trainer_cls(root, config)
    else:
        trainer = trainer_cls(root, config, run_dir)
    if optuna_trial is not None:
        trainer._optuna_trial = optuna_trial
    trainer.run()
    print(f"Saved run outputs -> {run_dir}")
    return run_dir


def print_models() -> None:
    for model_name, spec in MODEL_REGISTRY.items():
        print(f"- {model_name}: {spec.description}")


def optimize_experiment(
    model_name: str,
    root: Path,
    runs_root: str,
    optimizer_name: str,
    tag: str | None,
    fixed_overrides: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    metric_path: str,
    direction: str,
) -> Path:
    strategy = get_search_strategy(optimizer_name)
    try:
        optuna = importlib.import_module("optuna")
    except ImportError as exc:
        raise ImportError(
            "optuna is required for optimizer modes. Install it with `pip install optuna`."
        ) from exc

    base_name = build_run_name(f"{optimizer_name}-{model_name}", tag=tag)
    sweep_dir = root / runs_root / base_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = sweep_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    sampler = create_optuna_sampler(optuna, strategy)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=5
    )

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        sampled_overrides = sample_search_overrides(trial, search_space)
        merged_overrides = dict(fixed_overrides)
        merged_overrides.update(sampled_overrides)
        trial_tag = f"trial-{trial.number:03d}"
        print(
            f"Starting BO trial {trial.number + 1}/{n_trials} | "
            f"trial_id={trial.number:03d} | params={json.dumps(sampled_overrides, sort_keys=True)}"
        )
        trial_runs_root = path_relative_to_root_if_possible(trials_dir, root)
        run_dir = run_experiment(
            model_name=model_name,
            root=root,
            runs_root=str(trial_runs_root),
            tag=trial_tag,
            overrides=merged_overrides,
            optuna_trial=trial,
        )
        score = read_metrics_value(run_dir / "metrics.json", metric_path)
        trial.set_user_attr("run_dir", str(run_dir))
        trial_rows.append(
            {
                "trial": trial.number,
                "score": score,
                "run_dir": str(run_dir),
                **sampled_overrides,
            }
        )
        print(
            f"Finished BO trial {trial.number + 1}/{n_trials} | "
            f"score={score:.6g} | run_dir={run_dir}"
        )
        return score

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    summary = {
        "model": model_name,
        "optimizer": optimizer_name,
        "direction": direction,
        "metric_path": metric_path,
        "n_trials": n_trials,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_run_dir": study.best_trial.user_attrs.get("run_dir"),
        "search_space": search_space,
        "fixed_overrides": fixed_overrides,
    }
    (sweep_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_trials_csv(sweep_dir / "trials.csv", trial_rows)
    write_leaderboard(trials_dir, metric_path, direction)
    print(f"Saved optimization artifacts -> {sweep_dir}")
    return sweep_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab-friendly SMA training framework with selectable models."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root",
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_REGISTRY), default="ann", help="Model to train"
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(SEARCH_STRATEGY_REGISTRY),
        default="none",
        help="Search strategy registry entry for single run or hyperparameter optimization",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="colab_runs",
        help="Directory for timestamped run outputs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run tag appended to the run name",
    )
    parser.add_argument(
        "--config-json", type=str, default=None, help="JSON string of config overrides"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to JSON file of config overrides",
    )
    parser.add_argument(
        "--search-space-json",
        type=str,
        default=None,
        help="JSON string overriding the optimizer search space",
    )
    parser.add_argument(
        "--search-space-file",
        type=Path,
        default=None,
        help="Path to JSON file overriding the optimizer search space",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12,
        help="Number of optimization trials when optimizer is enabled",
    )
    parser.add_argument(
        "--metric-path",
        type=str,
        default="val_metrics_norm.mean_rmse",
        help="Metrics.json path used as the optimization objective",
    )
    parser.add_argument(
        "--direction",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Optimization direction for the selected metric",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--summarize-runs-dir",
        type=Path,
        default=None,
        help="Generate leaderboard.csv/json for completed run folders under this directory and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print_models()
        return

    if args.summarize_runs_dir is not None:
        leaderboard_file = write_leaderboard(
            args.summarize_runs_dir.resolve(), args.metric_path, args.direction
        )
        print(f"Saved leaderboard -> {leaderboard_file}")
        return

    overrides = load_overrides(args.config_json, args.config_file)
    if args.optimizer == "none":
        run_experiment(
            model_name=args.model,
            root=args.root.resolve(),
            runs_root=args.runs_root,
            tag=args.tag,
            overrides=overrides,
        )
        return

    search_space = load_search_space(
        args.model, args.search_space_json, args.search_space_file
    )
    optimize_experiment(
        model_name=args.model,
        root=args.root.resolve(),
        runs_root=args.runs_root,
        optimizer_name=args.optimizer,
        tag=args.tag,
        fixed_overrides=overrides,
        search_space=search_space,
        n_trials=args.n_trials,
        metric_path=args.metric_path,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()
