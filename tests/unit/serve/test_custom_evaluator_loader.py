"""Unit tests for custom evaluator function loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleServeError
from serve.custom_evaluator_loader import load_custom_evaluator


def test_load_valid_evaluator(tmp_path: Path) -> None:
    """Valid evaluator file with evaluate() function should load."""
    evaluator_file = tmp_path / "my_evaluator.py"
    evaluator_file.write_text(
        "def evaluate(model, dataset, device):\n"
        "    return {'accuracy': 0.95, 'loss': 0.05}\n",
        encoding="utf-8",
    )

    evaluator_fn = load_custom_evaluator(str(evaluator_file))

    assert callable(evaluator_fn)
    result = evaluator_fn(None, None, "cpu")
    assert result == {"accuracy": 0.95, "loss": 0.05}


def test_load_evaluator_not_callable(tmp_path: Path) -> None:
    """File with evaluate that is not callable should raise CrucibleServeError."""
    evaluator_file = tmp_path / "bad_evaluator.py"
    evaluator_file.write_text(
        "evaluate = 'not a function'\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleServeError, match="not callable"):
        load_custom_evaluator(str(evaluator_file))


def test_load_evaluator_missing_file() -> None:
    """Non-existent path should raise CrucibleServeError."""
    with pytest.raises(CrucibleServeError, match="Evaluator file not found"):
        load_custom_evaluator("/tmp/does_not_exist_evaluator_abc123.py")


def test_load_evaluator_no_function(tmp_path: Path) -> None:
    """File without evaluate function should raise CrucibleServeError."""
    evaluator_file = tmp_path / "no_eval.py"
    evaluator_file.write_text(
        "def some_other_function():\n"
        "    return 42\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleServeError, match="missing function 'evaluate'"):
        load_custom_evaluator(str(evaluator_file))
