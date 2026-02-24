"""Unit tests for compute type definitions."""

from __future__ import annotations

import pytest

from core.compute_types import (
    ComputeTarget,
    ConnectionProfile,
    JobStatus,
    JobSubmission,
)


def test_compute_target_defaults() -> None:
    """ComputeTarget should default executor_type to 'local'."""
    target = ComputeTarget(name="my-machine")
    assert target.name == "my-machine"
    assert target.executor_type == "local"


def test_job_status_is_frozen() -> None:
    """JobStatus should be immutable after construction."""
    status = JobStatus(job_id="abc123", state="queued")
    with pytest.raises(AttributeError):
        status.state = "running"  # type: ignore[misc]


def test_job_submission_optional_defaults() -> None:
    """JobSubmission optional fields should default correctly."""
    target = ComputeTarget(name="local")
    submission = JobSubmission(target=target, command="echo")
    assert submission.args == ()
    assert submission.working_dir is None
    assert submission.env_vars is None
    assert submission.command == "echo"


def test_connection_profile_fields() -> None:
    """ConnectionProfile should expose all expected fields."""
    profile = ConnectionProfile(
        target_name="gpu-box",
        connected=True,
        latency_ms=12.5,
    )
    assert profile.target_name == "gpu-box"
    assert profile.connected is True
    assert profile.latency_ms == 12.5
