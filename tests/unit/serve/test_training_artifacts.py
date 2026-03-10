"""Unit tests for training artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import CrucibleDependencyError
from core.types import BatchLossMetric, EpochMetric
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_training_history,
    save_training_plot,
)


def test_ensure_training_output_dir_creates_directory(tmp_path: Path) -> None:
    """Output directory helper should create missing directory."""
    output_dir = ensure_training_output_dir(str(tmp_path / "artifacts"))

    assert output_dir.exists()


def test_save_training_history_writes_json_file(tmp_path: Path) -> None:
    """History writer should persist history JSON file."""
    history_path = save_training_history(
        tmp_path,
        [EpochMetric(epoch=1, train_loss=1.2, validation_loss=1.3)],
        [BatchLossMetric(epoch=1, batch_index=1, global_step=1, train_loss=1.2)],
    )

    assert history_path.exists()


def test_save_training_history_writes_batch_loss_rows(tmp_path: Path) -> None:
    """History writer should include batch-level loss rows."""
    history_path = save_training_history(
        tmp_path,
        [EpochMetric(epoch=1, train_loss=1.2, validation_loss=1.3)],
        [BatchLossMetric(epoch=1, batch_index=1, global_step=1, train_loss=1.2)],
    )

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(payload["batch_losses"]) == 1


def test_save_training_plot_raises_without_matplotlib(monkeypatch, tmp_path: Path) -> None:
    """Plot writer should raise dependency error when matplotlib is unavailable."""
    import builtins

    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    with pytest.raises(CrucibleDependencyError):
        save_training_plot(
            tmp_path,
            [EpochMetric(epoch=1, train_loss=1.0, validation_loss=1.0)],
            [BatchLossMetric(epoch=1, batch_index=1, global_step=1, train_loss=1.0)],
        )

    assert True
