"""Unit tests for the LocalRunner execution backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.job_types import JobSpec, ResourceConfig
from serve.local_runner import LocalRunner
from store.job_store import load_job, list_jobs


@dataclass
class FakeTrainingResult:
    model_path: Path | None = None


def test_local_runner_kind() -> None:
    runner = LocalRunner()
    assert runner.kind == "local"


def test_submit_creates_record_on_disk(tmp_path: Path) -> None:
    """submit() should write initial + final job records."""
    runner = LocalRunner()
    spec = JobSpec(job_type="sft", method_args={"--epochs": "1"}, label="Test")

    fake_result = FakeTrainingResult(model_path=Path("/tmp/model.pt"))
    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    assert record.state == "completed"
    assert record.model_path == "/tmp/model.pt"
    assert record.job_type == "sft"
    assert record.label == "Test"
    assert record.backend == "local"
    # Record persisted to disk
    loaded = load_job(tmp_path, record.job_id)
    assert loaded.state == "completed"
    assert loaded.model_path == "/tmp/model.pt"


def test_submit_marks_failed_on_exception(tmp_path: Path) -> None:
    """submit() should mark record as failed if training throws."""
    runner = LocalRunner()
    spec = JobSpec(job_type="sft", method_args={})

    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", side_effect=RuntimeError("CUDA OOM")),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
        pytest.raises(RuntimeError, match="CUDA OOM"),
    ):
        runner.submit(tmp_path, spec)

    # Record was saved as failed
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "failed"
    assert "CUDA OOM" in jobs[0].error_message


def test_submit_handles_no_model_path(tmp_path: Path) -> None:
    """submit() should handle training result with no model_path."""
    runner = LocalRunner()
    spec = JobSpec(job_type="eval", method_args={})

    fake_result = FakeTrainingResult(model_path=None)
    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    assert record.state == "completed"
    assert record.model_path == ""


def test_cancel_updates_state(tmp_path: Path) -> None:
    runner = LocalRunner()
    spec = JobSpec(job_type="sft", method_args={})

    fake_result = FakeTrainingResult(model_path=Path("/tmp/model.pt"))
    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    cancelled = runner.cancel(tmp_path, record.job_id)
    assert cancelled.state == "cancelled"


def test_get_state_reads_from_disk(tmp_path: Path) -> None:
    runner = LocalRunner()
    spec = JobSpec(job_type="sft", method_args={})

    fake_result = FakeTrainingResult(model_path=Path("/tmp/model.pt"))
    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    state = runner.get_state(tmp_path, record.job_id)
    assert state == "completed"


def test_get_logs_returns_empty() -> None:
    runner = LocalRunner()
    assert runner.get_logs(Path("/tmp"), "any-id") == ""


def test_get_result_returns_dict(tmp_path: Path) -> None:
    runner = LocalRunner()
    spec = JobSpec(job_type="sft", method_args={})

    fake_result = FakeTrainingResult(model_path=Path("/tmp/model.pt"))
    fake_config = MagicMock(data_root=tmp_path)

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    result = runner.get_result(tmp_path, record.job_id)
    assert result["job_id"] == record.job_id
    assert result["state"] == "completed"
    assert result["model_path"] == "/tmp/model.pt"
