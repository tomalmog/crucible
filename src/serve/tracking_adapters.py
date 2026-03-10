"""External experiment tracking adapters for W&B and TensorBoard.

This module defines a protocol for tracking adapters and provides
concrete implementations for Weights & Biases and TensorBoard.
Adapters are optional — missing dependencies raise CrucibleDependencyError.
"""

from __future__ import annotations

from typing import Any, Protocol


class TrackingAdapter(Protocol):
    """Protocol for external experiment tracking backends."""

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log metric values at a training step."""
        ...

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameter configuration."""
        ...

    def finish(self) -> None:
        """Flush and close the tracking session."""
        ...


class WandbAdapter:
    """Weights & Biases experiment tracking adapter."""

    def __init__(self, project: str, run_name: str) -> None:
        from core.errors import CrucibleDependencyError
        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError as error:
            raise CrucibleDependencyError(
                "W&B tracking requires the wandb package. "
                "Install with: pip install wandb"
            ) from error
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics to W&B."""
        self._run.log(metrics, step=step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters as W&B config."""
        self._run.config.update(params)

    def finish(self) -> None:
        """Finish the W&B run."""
        self._run.finish()


class TensorBoardAdapter:
    """TensorBoard experiment tracking adapter."""

    def __init__(self, log_dir: str) -> None:
        from core.errors import CrucibleDependencyError
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
        except ImportError as error:
            raise CrucibleDependencyError(
                "TensorBoard tracking requires tensorboard. "
                "Install with: pip install tensorboard"
            ) from error
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar metrics to TensorBoard."""
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters as TensorBoard text."""
        text = "\n".join(f"**{k}**: {v}" for k, v in params.items())
        self._writer.add_text("hyperparameters", text)

    def finish(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.flush()
        self._writer.close()


def build_tracking_adapters(
    wandb_project: str | None,
    tensorboard_dir: str | None,
    run_name: str = "",
) -> list[TrackingAdapter]:
    """Build a list of active tracking adapters from configuration."""
    adapters: list[TrackingAdapter] = []
    if wandb_project:
        adapters.append(WandbAdapter(project=wandb_project, run_name=run_name))
    if tensorboard_dir:
        adapters.append(TensorBoardAdapter(log_dir=tensorboard_dir))
    return adapters
