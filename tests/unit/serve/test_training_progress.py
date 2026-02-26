"""Unit tests for training progress reporting helpers."""

from __future__ import annotations

import json

from serve.training_progress import TrainingProgressTracker, read_optimizer_learning_rate


class _FakeOptimizer:
    def __init__(self, lr: float) -> None:
        self.param_groups = [{"lr": lr}]


def test_training_progress_tracker_logs_periodic_batch_updates(capsys) -> None:
    """Progress tracker should emit first/interval/last batch updates."""
    tracker = TrainingProgressTracker(
        dataset_name="demo",
        total_epochs=3,
        start_epoch=1,
        train_batch_count=5,
        validation_batch_count=2,
        batch_log_interval_steps=2,
    )

    tracker.log_training_started()
    tracker.log_epoch_started(1)
    for batch_index in [1, 2, 3, 4, 5]:
        tracker.log_batch_progress("train", 1, batch_index, 5, batch_index, 0.5)
    tracker.log_epoch_completed(1, train_loss=0.5, validation_loss=0.4, learning_rate=0.001)

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    events = [json.loads(line) for line in lines]
    batch_events = [e for e in events if e["event"] == "training_batch_progress"]

    assert len(batch_events) == 4
    assert all(e["event"] == "training_batch_progress" for e in batch_events)


def test_read_optimizer_learning_rate_reads_first_param_group() -> None:
    """Learning-rate reader should return the first optimizer param-group value."""
    learning_rate = read_optimizer_learning_rate(_FakeOptimizer(lr=0.003))

    assert learning_rate == 0.003
