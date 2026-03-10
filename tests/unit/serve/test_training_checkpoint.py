"""Unit tests for training checkpoint persistence helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import pytest

from core.errors import CrucibleServeError
from serve.training_checkpoint import (
    ensure_checkpoint_dir,
    load_resume_checkpoint,
    prune_epoch_checkpoints,
    save_best_checkpoint,
    save_epoch_checkpoint,
)


class _FakeTorch:
    def save(self, payload: object, path: str) -> None:
        Path(path).write_bytes(pickle.dumps(payload))

    def load(self, path: str, map_location: object) -> object:
        _ = map_location
        return pickle.loads(Path(path).read_bytes())


class _FakeModel:
    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = dict(state)
        self.loaded_state: Mapping[str, object] | None = None

    def state_dict(self) -> Mapping[str, object]:
        return self._state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.loaded_state = dict(state)


class _FakeOptimizer:
    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = dict(state)
        self.loaded_state: Mapping[str, object] | None = None

    def state_dict(self) -> Mapping[str, object]:
        return self._state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.loaded_state = dict(state)


class _FakeScheduler:
    def __init__(self, state: Mapping[str, object]) -> None:
        self._state = dict(state)
        self.loaded_state: Mapping[str, object] | None = None

    def state_dict(self) -> Mapping[str, object]:
        return self._state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.loaded_state = dict(state)


def test_save_epoch_checkpoint_and_prune_keeps_latest_files(tmp_path: Path) -> None:
    """Pruning should keep only the newest epoch checkpoint files."""
    torch_module = _FakeTorch()
    model = _FakeModel(state={"weight": 1})
    optimizer = _FakeOptimizer(state={"step": 1})
    checkpoint_dir = ensure_checkpoint_dir(tmp_path)
    for epoch in [1, 2, 3]:
        save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            torch_module=torch_module,
            model=model,
            optimizer=optimizer,
            scheduler=_FakeScheduler(state={"last_epoch": epoch}),
            epoch=epoch,
            global_step=epoch * 10,
            best_validation_loss=0.7,
        )
    prune_epoch_checkpoints(checkpoint_dir, max_files=2)

    assert [path.name for path in sorted(checkpoint_dir.glob("epoch-*.pt"))] == [
        "epoch-0002.pt",
        "epoch-0003.pt",
    ]


def test_load_resume_checkpoint_restores_model_and_optimizer(tmp_path: Path) -> None:
    """Resume loading should restore model and optimizer state with metadata."""
    torch_module = _FakeTorch()
    checkpoint_dir = ensure_checkpoint_dir(tmp_path)
    save_best_checkpoint(
        checkpoint_dir=checkpoint_dir,
        torch_module=torch_module,
        model=_FakeModel(state={"weight": 42}),
        optimizer=_FakeOptimizer(state={"momentum": 5}),
        scheduler=_FakeScheduler(state={"last_epoch": 4}),
        epoch=4,
        global_step=32,
        best_validation_loss=0.123,
    )
    model = _FakeModel(state={"weight": 0})
    optimizer = _FakeOptimizer(state={"momentum": 0})
    scheduler = _FakeScheduler(state={"last_epoch": 0})

    state = load_resume_checkpoint(
        checkpoint_path=str(checkpoint_dir / "best.pt"),
        torch_module=torch_module,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )

    assert (
        state.next_epoch == 5
        and state.global_step == 32
        and state.best_validation_loss == 0.123
        and model.loaded_state == {"weight": 42}
        and optimizer.loaded_state == {"momentum": 5}
        and scheduler.loaded_state == {"last_epoch": 4}
    )


def test_load_resume_checkpoint_raises_for_missing_file(tmp_path: Path) -> None:
    """Resume loading should fail with a clear error for missing files."""
    with pytest.raises(CrucibleServeError):
        load_resume_checkpoint(
            checkpoint_path=str(tmp_path / "missing.pt"),
            torch_module=_FakeTorch(),
            model=_FakeModel(state={}),
            optimizer=_FakeOptimizer(state={}),
            scheduler=None,
            device="cpu",
        )

    assert True


def test_load_resume_checkpoint_raises_for_invalid_payload(tmp_path: Path) -> None:
    """Resume loading should reject checkpoint payloads without required keys."""
    checkpoint_path = tmp_path / "invalid.pt"
    _FakeTorch().save(payload={"epoch": "bad"}, path=str(checkpoint_path))

    with pytest.raises(CrucibleServeError):
        load_resume_checkpoint(
            checkpoint_path=str(checkpoint_path),
            torch_module=_FakeTorch(),
            model=_FakeModel(state={}),
            optimizer=_FakeOptimizer(state={}),
            scheduler=None,
            device="cpu",
        )

    assert True
