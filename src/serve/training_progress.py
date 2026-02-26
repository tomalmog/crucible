"""Structured training progress reporting.

This module emits periodic progress events for long-running training jobs,
including batch-level updates, epoch summaries, and ETA estimates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


def emit_progress(event: str, **fields: object) -> None:
    """Write a JSON progress line to stdout, flushed immediately."""
    payload = {"event": event, **fields}
    print(json.dumps(payload), flush=True)


@dataclass
class TrainingProgressTracker:
    """Track and emit training progress events across epochs."""

    dataset_name: str
    total_epochs: int
    start_epoch: int
    train_batch_count: int
    validation_batch_count: int
    batch_log_interval_steps: int
    run_started_at: float = field(default_factory=time.monotonic)
    current_epoch_started_at: float | None = None

    def log_training_started(self) -> None:
        """Log one event when a training run starts."""
        emit_progress(
            "training_started",
            dataset_name=self.dataset_name,
            total_epochs=self.total_epochs,
            start_epoch=self.start_epoch,
            train_batches=self.train_batch_count,
            validation_batches=self.validation_batch_count,
        )

    def log_epoch_started(self, epoch_index: int) -> None:
        """Log epoch start and capture epoch timer state."""
        self.current_epoch_started_at = time.monotonic()
        emit_progress(
            "training_epoch_started",
            dataset_name=self.dataset_name,
            epoch=epoch_index,
            total_epochs=self.total_epochs,
        )

    def log_batch_progress(
        self,
        phase: str,
        epoch_index: int,
        batch_index: int,
        total_batches: int,
        global_step: int,
        loss: float,
    ) -> None:
        """Log periodic batch progress updates during one epoch."""
        if not _should_log_batch(batch_index, total_batches, self.batch_log_interval_steps):
            return
        progress_fraction = _progress_fraction(batch_index, total_batches)
        emit_progress(
            "training_batch_progress",
            dataset_name=self.dataset_name,
            phase=phase,
            epoch=epoch_index,
            total_epochs=self.total_epochs,
            batch=batch_index,
            total_batches=total_batches,
            epoch_progress=round(progress_fraction, 3),
            global_step=global_step,
            loss=round(loss, 6),
        )

    def log_epoch_completed(
        self,
        epoch_index: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
    ) -> None:
        """Log epoch completion summary with elapsed and ETA estimates."""
        now = time.monotonic()
        epoch_elapsed_seconds = _elapsed_seconds(self.current_epoch_started_at, now)
        run_elapsed_seconds = now - self.run_started_at
        completed_epochs = max(1, epoch_index - self.start_epoch + 1)
        remaining_epochs = max(0, self.total_epochs - epoch_index)
        eta_seconds = (run_elapsed_seconds / completed_epochs) * remaining_epochs
        emit_progress(
            "training_epoch_completed",
            dataset_name=self.dataset_name,
            epoch=epoch_index,
            total_epochs=self.total_epochs,
            train_loss=round(train_loss, 6),
            validation_loss=round(validation_loss, 6),
            learning_rate=round(learning_rate, 10),
            epoch_elapsed_seconds=round(epoch_elapsed_seconds, 3),
            run_elapsed_seconds=round(run_elapsed_seconds, 3),
            eta_seconds=round(eta_seconds, 3),
        )


def _should_log_batch(batch_index: int, total_batches: int, interval: int) -> bool:
    """Return true when the current batch should emit a progress event."""
    if batch_index <= 1 or batch_index >= total_batches:
        return True
    return batch_index % interval == 0


def _progress_fraction(batch_index: int, total_batches: int) -> float:
    """Compute bounded in-epoch progress fraction."""
    if total_batches <= 0:
        return 0.0
    return min(1.0, max(0.0, batch_index / total_batches))


def _elapsed_seconds(start_at: float | None, end_at: float) -> float:
    """Compute elapsed seconds, handling unset start timestamps."""
    if start_at is None:
        return 0.0
    return max(0.0, end_at - start_at)


def read_optimizer_learning_rate(optimizer: Any) -> float:
    """Read current learning rate from the first optimizer param group."""
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list) or not param_groups:
        return 0.0
    first_group = param_groups[0]
    if not isinstance(first_group, dict):
        return 0.0
    learning_rate = first_group.get("lr")
    if isinstance(learning_rate, (int, float)):
        return float(learning_rate)
    return 0.0
