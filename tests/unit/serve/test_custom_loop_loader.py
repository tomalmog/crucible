"""Unit tests for custom loop loader behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleServeError
from serve.custom_loop_loader import load_custom_training_loop


def test_load_custom_training_loop_returns_none_without_path() -> None:
    """Omitting custom loop path should return None."""
    loop = load_custom_training_loop(None)

    assert loop is None


def test_load_custom_training_loop_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing custom loop file should raise serve error."""
    with pytest.raises(CrucibleServeError):
        load_custom_training_loop(str(tmp_path / "missing.py"))

    assert True


def test_load_custom_training_loop_loads_callable(tmp_path: Path) -> None:
    """Valid custom loop file should load callable function."""
    custom_loop_path = tmp_path / "loop.py"
    custom_loop_path.write_text(
        "from core.types import EpochMetric\n"
        "def run_custom_training(context):\n"
        "    return [EpochMetric(epoch=1, train_loss=1.0, validation_loss=1.1)]\n",
        encoding="utf-8",
    )

    loop = load_custom_training_loop(str(custom_loop_path))

    assert callable(loop)
