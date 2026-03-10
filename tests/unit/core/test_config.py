"""Unit tests for core config parsing."""

from __future__ import annotations

import os

import pytest

from core.config import CrucibleConfig
from core.errors import CrucibleConfigError


def test_from_env_reads_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should resolve data root from environment."""
    monkeypatch.setenv("CRUCIBLE_DATA_ROOT", "./.tmp-crucible")

    config = CrucibleConfig.from_env()

    assert config.data_root.name == ".tmp-crucible"


def test_from_env_raises_for_invalid_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail for non-numeric random seed."""
    monkeypatch.setenv("CRUCIBLE_RANDOM_SEED", "not-a-number")

    with pytest.raises(CrucibleConfigError):
        CrucibleConfig.from_env()

    assert os.getenv("CRUCIBLE_RANDOM_SEED") == "not-a-number"
