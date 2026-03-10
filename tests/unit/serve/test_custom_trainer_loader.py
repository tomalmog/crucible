"""Unit tests for custom trainer class loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleServeError
from serve.custom_trainer_loader import load_custom_trainer


def test_load_valid_trainer_class(tmp_path: Path) -> None:
    """Valid trainer file with Trainer class and train method should load."""
    trainer_file = tmp_path / "my_trainer.py"
    trainer_file.write_text(
        "class Trainer:\n"
        "    def train(self, context):\n"
        "        return context\n",
        encoding="utf-8",
    )

    trainer_cls = load_custom_trainer(str(trainer_file))

    assert trainer_cls is not None
    assert hasattr(trainer_cls, "train")
    assert callable(trainer_cls.train)


def test_load_trainer_missing_train_method(tmp_path: Path) -> None:
    """Trainer class without train() method should raise CrucibleServeError."""
    trainer_file = tmp_path / "bad_trainer.py"
    trainer_file.write_text(
        "class Trainer:\n"
        "    def predict(self, x):\n"
        "        return x\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleServeError, match="missing a callable 'train' method"):
        load_custom_trainer(str(trainer_file))


def test_load_trainer_missing_file() -> None:
    """Non-existent path should raise CrucibleServeError."""
    with pytest.raises(CrucibleServeError, match="Trainer file not found"):
        load_custom_trainer("/tmp/does_not_exist_trainer_abc123.py")


def test_load_trainer_no_class(tmp_path: Path) -> None:
    """File without Trainer class should raise CrucibleServeError."""
    trainer_file = tmp_path / "no_class.py"
    trainer_file.write_text(
        "def some_function():\n"
        "    return 42\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleServeError, match="missing class 'Trainer'"):
        load_custom_trainer(str(trainer_file))
