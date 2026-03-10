"""Training artifacts persistence and graph generation.

This module writes training histories and produces standard curves.
It captures epoch loss metrics for train and validation phases.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.constants import (
    DEFAULT_TRAIN_HISTORY_FILE_NAME,
    DEFAULT_TRAIN_PLOT_FILE_NAME,
    DEFAULT_TRAINED_MODEL_FILE_NAME,
)
from core.errors import CrucibleDependencyError
from core.types import BatchLossMetric, EpochMetric


def ensure_training_output_dir(output_dir: str) -> Path:
    """Create and return resolved training output directory.

    Args:
        output_dir: Configured output directory.

    Returns:
        Resolved output path.
    """
    resolved_path = Path(output_dir).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def save_training_history(
    output_dir: Path,
    metrics: list[EpochMetric],
    batch_losses: list[BatchLossMetric],
) -> Path:
    """Persist training history JSON file.

    Args:
        output_dir: Training output directory.
        metrics: Epoch metrics.
        batch_losses: Per-batch training loss metrics.

    Returns:
        History JSON file path.
    """
    history_path = output_dir / DEFAULT_TRAIN_HISTORY_FILE_NAME
    payload = {
        "epochs": [asdict(metric) for metric in metrics],
        "batch_losses": [asdict(metric) for metric in batch_losses],
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return history_path


def save_model_weights(output_dir: Path, torch_module: Any, model: Any) -> Path:
    """Persist model state dict to output directory.

    Args:
        output_dir: Training output directory.
        torch_module: Imported torch module.
        model: Trained torch model.

    Returns:
        Model file path.
    """
    model_path = output_dir / DEFAULT_TRAINED_MODEL_FILE_NAME
    torch_module.save(model.state_dict(), str(model_path))
    return model_path


def save_training_plot(
    output_dir: Path,
    metrics: list[EpochMetric],
    batch_losses: list[BatchLossMetric],
) -> Path | None:
    """Save loss curves as PNG when matplotlib is available.

    Args:
        output_dir: Training output directory.
        metrics: Epoch metrics.
        batch_losses: Per-batch training loss metrics.

    Returns:
        Plot file path when generated; otherwise None.

    Raises:
        CrucibleDependencyError: If matplotlib is missing.
    """
    if not metrics:
        return None
    try:
        import matplotlib.pyplot as plot
    except ImportError as error:
        raise CrucibleDependencyError(
            "Training plot generation requires matplotlib. "
            "Install matplotlib to produce loss graphs."
        ) from error
    figure = _build_plot_figure(plot, metrics, batch_losses)
    plot_path = output_dir / DEFAULT_TRAIN_PLOT_FILE_NAME
    figure.tight_layout()
    figure.savefig(plot_path)
    plot.close(figure)
    return plot_path


def _build_plot_figure(
    plot: Any, metrics: list[EpochMetric], batch_losses: list[BatchLossMetric]
) -> Any:
    """Build matplotlib figure with adaptive number of chart rows."""
    if batch_losses:
        figure, axes = plot.subplots(2, 1, figsize=(9, 7), sharex=False)
        _plot_batch_loss_axis(axes[0], batch_losses)
        _plot_epoch_loss_axis(axes[1], metrics)
        return figure
    figure, axis = plot.subplots(1, 1, figsize=(9, 4.8))
    _plot_epoch_loss_axis(axis, metrics)
    return figure


def _plot_batch_loss_axis(axis: Any, batch_losses: list[BatchLossMetric]) -> None:
    """Render detailed per-step training loss curve."""
    steps = [metric.global_step for metric in batch_losses]
    losses = [metric.train_loss for metric in batch_losses]
    axis.plot(steps, losses, color="#0c8e7c", linewidth=1.8, label="train_loss_step")
    axis.set_title("Training Loss by Step")
    axis.set_xlabel("Global Step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()


def _plot_epoch_loss_axis(axis: Any, metrics: list[EpochMetric]) -> None:
    """Render epoch train/validation loss curves."""
    epoch_indexes = [metric.epoch for metric in metrics]
    train_losses = [metric.train_loss for metric in metrics]
    validation_losses = [metric.validation_loss for metric in metrics]
    axis.plot(epoch_indexes, train_losses, color="#0c8e7c", linewidth=2.2, label="train_loss")
    axis.plot(
        epoch_indexes,
        validation_losses,
        color="#cf5f2f",
        linewidth=2.2,
        label="validation_loss",
    )
    axis.set_title("Epoch Loss Curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()
