from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_NAMES = ["C", "L", "k", "Asd"]


class AnnResultVisualizer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.metrics_path = output_dir / "metrics.json"
        self.predictions_path = output_dir / "predictions.npz"
        self.figure_dir = output_dir / "visualization_report"
        self.figure_dir.mkdir(exist_ok=True)

    def run(self) -> None:
        metrics = json.loads(self.metrics_path.read_text(encoding="ascii"))
        predictions = np.load(self.predictions_path)
        label_names = metrics.get("config", {}).get("label_names", LABEL_NAMES)

        self.save_metric_bar_chart(metrics, label_names)
        self.save_split_scatter_grid(predictions, label_names)
        self.save_residual_histograms(predictions, label_names)
        self.save_error_by_rate(predictions, label_names)
        self.save_summary_text(metrics, label_names)

        print(f"Saved visualization report -> {self.figure_dir}")

    def save_metric_bar_chart(self, metrics: dict, label_names: list[str]) -> None:
        splits = ["train", "val", "test"]
        metric_names = [
            ("rmse", "physical"),
            ("rmse", "normalized"),
            ("r2", "physical"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        x = np.arange(len(label_names))
        width = 0.25

        for ax, (metric_name, scale_name) in zip(axes, metric_names):
            for idx, split in enumerate(splits):
                metric_key = f"{split}_metrics"
                if scale_name == "normalized":
                    metric_key = f"{metric_key}_norm"
                values = np.asarray(metrics[metric_key][metric_name], dtype=float)
                ax.bar(x + (idx - 1) * width, values, width=width, label=split)
            ax.set_xticks(x)
            ax.set_xticklabels(label_names)
            ax.set_title(f"{metric_name.upper()} ({scale_name})")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("Value")
        axes[-1].legend()
        fig.suptitle("Regression Metrics by Split")
        fig.tight_layout()
        fig.savefig(self.figure_dir / "metrics_bar_chart.png", dpi=170)
        plt.close(fig)

    def save_split_scatter_grid(self, predictions: np.lib.npyio.NpzFile, label_names: list[str]) -> None:
        split_specs = [
            ("train", predictions["y_train_true"], predictions["y_train_pred"]),
            ("val", predictions["y_val_true"], predictions["y_val_pred"]),
            ("test", predictions["y_test_true"], predictions["y_test_pred"]),
        ]

        n_outputs = len(label_names)
        fig, axes = plt.subplots(3, n_outputs, figsize=(4 * n_outputs, 11), squeeze=False)
        for row_idx, (split_name, y_true, y_pred) in enumerate(split_specs):
            for col_idx, label in enumerate(label_names):
                axis = axes[row_idx, col_idx]
                axis.scatter(y_true[:, col_idx], y_pred[:, col_idx], s=9, alpha=0.55)
                lower = float(min(np.min(y_true[:, col_idx]), np.min(y_pred[:, col_idx])))
                upper = float(max(np.max(y_true[:, col_idx]), np.max(y_pred[:, col_idx])))
                axis.plot([lower, upper], [lower, upper], "r--", linewidth=1.0)
                if row_idx == 0:
                    axis.set_title(label)
                if col_idx == 0:
                    axis.set_ylabel(f"{split_name} pred")
                axis.set_xlabel("true")
                axis.grid(True, alpha=0.25)
        fig.suptitle("True vs Predicted Values by Split")
        fig.tight_layout()
        fig.savefig(self.figure_dir / "split_scatter_grid.png", dpi=170)
        plt.close(fig)

    def save_residual_histograms(self, predictions: np.lib.npyio.NpzFile, label_names: list[str]) -> None:
        residuals = predictions["y_test_pred"] - predictions["y_test_true"]
        n_outputs = len(label_names)
        n_cols = 2
        n_rows = int(np.ceil(n_outputs / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows), squeeze=False)
        flat_axes = axes.reshape(-1)
        for idx, axis in enumerate(flat_axes[:n_outputs]):
            axis.hist(residuals[:, idx], bins=30, color="#4C78A8", edgecolor="black", alpha=0.85)
            axis.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
            axis.set_title(f"Test Residuals: {label_names[idx]}")
            axis.set_xlabel("pred - true")
            axis.set_ylabel("Count")
            axis.grid(True, alpha=0.25)
        for axis in flat_axes[n_outputs:]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(self.figure_dir / "test_residual_histograms.png", dpi=170)
        plt.close(fig)

    def save_error_by_rate(self, predictions: np.lib.npyio.NpzFile, label_names: list[str]) -> None:
        rates = predictions["rate_test"].reshape(-1)
        abs_errors = np.abs(predictions["y_test_pred"] - predictions["y_test_true"])
        unique_rates = np.unique(rates)
        error_matrix = np.zeros((unique_rates.size, abs_errors.shape[1]), dtype=float)

        for idx, rate in enumerate(unique_rates):
            mask = np.abs(rates - rate) < 1e-6
            error_matrix[idx, :] = np.mean(abs_errors[mask], axis=0)

        counts = np.array([np.sum(np.abs(rates - rate) < 1e-6) for rate in unique_rates], dtype=int)

        n_outputs = len(label_names)
        n_cols = 2
        n_rows = int(np.ceil(n_outputs / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows), squeeze=False)
        flat_axes = axes.reshape(-1)
        for idx, axis in enumerate(flat_axes[:n_outputs]):
            axis.bar(np.arange(unique_rates.size), error_matrix[:, idx], color="#72B7B2")
            axis.set_xticks(np.arange(unique_rates.size))
            axis.set_xticklabels([f"{rate:.2f}\n(n={count})" for rate, count in zip(unique_rates, counts)], rotation=25)
            axis.set_title(f"Test MAE by Rate: {label_names[idx]}")
            axis.set_xlabel("rate (%/s)")
            axis.set_ylabel("MAE")
            axis.grid(True, axis="y", alpha=0.25)
        for axis in flat_axes[n_outputs:]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(self.figure_dir / "test_error_by_rate.png", dpi=170)
        plt.close(fig)

    def save_summary_text(self, metrics: dict, label_names: list[str]) -> None:
        lines: list[str] = []
        for split_name in ("train_metrics", "val_metrics", "test_metrics"):
            lines.append(split_name)
            split_metrics = metrics[split_name]
            for metric_name in ("mae", "rmse", "r2"):
                values = ", ".join(
                    f"{label}={value:.6g}" for label, value in zip(label_names, split_metrics[metric_name])
                )
                lines.append(f"  {metric_name}: {values}")
            lines.append("")
        (self.figure_dir / "summary.txt").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved ANN training results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python_dnn_outputs"),
        help="Directory containing metrics.json and predictions.npz",
    )
    args = parser.parse_args()

    visualizer = AnnResultVisualizer(args.output_dir.resolve())
    visualizer.run()


if __name__ == "__main__":
    main()
