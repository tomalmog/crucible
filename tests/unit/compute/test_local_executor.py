"""Unit tests for local subprocess executor."""

from __future__ import annotations

import time

import pytest

from compute.local_executor import LocalExecutor
from core.compute_types import ComputeTarget, JobSubmission
from core.errors import CrucibleComputeError


def _make_submission(command: str, args: tuple[str, ...] = ()) -> JobSubmission:
    """Build a simple local job submission."""
    target = ComputeTarget(name="local")
    return JobSubmission(target=target, command=command, args=args)


def test_submit_returns_job_id() -> None:
    """submit() should return a non-empty string job_id."""
    executor = LocalExecutor()
    submission = _make_submission("echo", ("hello",))
    job_id = executor.submit(submission)
    assert isinstance(job_id, str)
    assert len(job_id) > 0


def test_status_completed_after_echo() -> None:
    """A simple echo command should complete with exit code 0."""
    executor = LocalExecutor()
    submission = _make_submission("echo", ("hello",))
    job_id = executor.submit(submission)
    # Give subprocess a moment to finish
    time.sleep(0.5)
    status = executor.status(job_id)
    assert status.state == "completed"
    assert status.exit_code == 0
    assert status.started_at is not None


def test_cancel_running_job() -> None:
    """cancel() should terminate a long-running process."""
    executor = LocalExecutor()
    submission = _make_submission("sleep", ("30",))
    job_id = executor.submit(submission)
    cancelled = executor.cancel(job_id)
    assert cancelled is True
    status = executor.status(job_id)
    assert status.state == "failed"


def test_status_unknown_job_raises() -> None:
    """status() should raise CrucibleComputeError for unknown job ids."""
    executor = LocalExecutor()
    with pytest.raises(CrucibleComputeError, match="Unknown job id"):
        executor.status("nonexistent-id")
