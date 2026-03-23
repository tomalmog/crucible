"""Unit tests for backend registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.backend_registry import get_backend, register_backend, _BACKENDS
from core.errors import CrucibleError
from core.job_types import BackendKind, JobRecord, JobSpec, JobState


class FakeBackend:
    """Minimal backend for testing."""

    @property
    def kind(self) -> BackendKind:
        return "local"

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord:
        raise NotImplementedError

    def cancel(self, data_root: Path, job_id: str) -> JobRecord:
        raise NotImplementedError

    def get_state(self, data_root: Path, job_id: str) -> JobState:
        return "running"

    def get_logs(self, data_root: Path, job_id: str, tail: int = 200) -> str:
        return ""

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]:
        return {}


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Clear registry before each test."""
    _BACKENDS.clear()


def test_register_and_get() -> None:
    backend = FakeBackend()
    register_backend("local", backend)
    assert get_backend("local") is backend


def test_get_unknown_raises() -> None:
    with pytest.raises(CrucibleError, match="Unknown backend"):
        get_backend("local")


def test_get_unknown_lists_available() -> None:
    register_backend("local", FakeBackend())
    try:
        get_backend("slurm")
    except CrucibleError as e:
        assert "local" in str(e)


def test_register_overwrites() -> None:
    b1 = FakeBackend()
    b2 = FakeBackend()
    register_backend("local", b1)
    register_backend("local", b2)
    assert get_backend("local") is b2
