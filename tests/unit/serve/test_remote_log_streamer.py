"""Unit tests for remote Slurm log streaming and state queries."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.slurm_types import ClusterConfig, RemoteJobRecord
from serve.remote_job_state import (
    check_remote_job_state,
    is_job_done,
    slurm_state_to_crucible,
)
from serve.remote_log_streamer import fetch_remote_logs

_LOG_MODULE = "serve.remote_log_streamer"
_STATE_MODULE = "serve.remote_job_state"


def _make_session(responses: list[tuple[str, str, int]]) -> MagicMock:
    session = MagicMock()
    session.execute = MagicMock(side_effect=responses)
    session.tail_last = MagicMock(return_value="log line 1\nlog line 2\n")
    return session


def _make_record(**overrides: object) -> RemoteJobRecord:
    defaults: dict[str, object] = dict(
        job_id="rj-test123",
        slurm_job_id="12345",
        cluster_name="test-hpc",
        training_method="sft",
        state="running",
        submitted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        remote_output_dir="/scratch/crucible/rj-test123",
        remote_log_path="/scratch/crucible/rj-test123/slurm-12345.out",
    )
    defaults.update(overrides)
    return RemoteJobRecord(**defaults)  # type: ignore[arg-type]


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(name="test-hpc", host="hpc.example.com", user="jdoe")


# -- slurm_state_to_crucible -----------------------------------------------------


def test_slurm_state_to_crucible_completed_returns_completed() -> None:
    """COMPLETED maps to completed."""
    assert slurm_state_to_crucible("COMPLETED") == "completed"


def test_slurm_state_to_crucible_failed_returns_failed() -> None:
    """FAILED maps to failed."""
    assert slurm_state_to_crucible("FAILED") == "failed"


def test_slurm_state_to_crucible_cancelled_returns_cancelled() -> None:
    """CANCELLED maps to cancelled."""
    assert slurm_state_to_crucible("CANCELLED") == "cancelled"


def test_slurm_state_to_crucible_unknown_defaults_running() -> None:
    """Unknown Slurm state defaults to running."""
    assert slurm_state_to_crucible("CONFIGURING") == "running"


# -- is_job_done ---------------------------------------------------------------


def test_is_job_done_true_for_completed() -> None:
    """sacct returning COMPLETED means the job is done."""
    session = _make_session([("COMPLETED\n", "", 0)])
    assert is_job_done(session, "12345") is True


def test_is_job_done_false_for_running() -> None:
    """sacct returning RUNNING means the job is not done."""
    session = _make_session([("RUNNING\n", "", 0)])
    assert is_job_done(session, "12345") is False


def test_is_job_done_false_on_sacct_error() -> None:
    """Non-zero exit code from sacct returns False."""
    session = _make_session([("", "error", 1)])
    assert is_job_done(session, "12345") is False


# -- fetch_remote_logs ---------------------------------------------------------


@patch(f"{_LOG_MODULE}.load_cluster", return_value=_make_cluster())
@patch(f"{_LOG_MODULE}.load_remote_job")
def test_fetch_remote_logs_returns_tail(
    mock_load_job: MagicMock,
    mock_load_cluster: MagicMock,
) -> None:
    """When log file exists, tail_last output is returned."""
    mock_load_job.return_value = _make_record()
    mock_session = _make_session([("", "", 0)])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_session)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch(f"{_LOG_MODULE}.SshSession", return_value=ctx):
        result = fetch_remote_logs(Path("/data"), "rj-test123")
    assert result == "log line 1\nlog line 2\n"


@patch(f"{_LOG_MODULE}.load_cluster", return_value=_make_cluster())
@patch(f"{_LOG_MODULE}.load_remote_job")
def test_fetch_remote_logs_returns_sacct_when_no_file(
    mock_load_job: MagicMock,
    mock_load_cluster: MagicMock,
) -> None:
    """When log file missing, sacct info is included in the message."""
    mock_load_job.return_value = _make_record(state="failed")
    mock_session = _make_session([
        ("", "", 1),  # test -f fails
        ("12345|FAILED|1:0||00:05:00|node01|OOM\n", "", 0),  # sacct query
    ])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_session)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch(f"{_LOG_MODULE}.SshSession", return_value=ctx):
        result = fetch_remote_logs(Path("/data"), "rj-test123")
    assert "Log file not found" in result
    assert "sacct" in result.lower() or "Slurm Job Info" in result


# -- check_remote_job_state ----------------------------------------------------


@patch(f"{_STATE_MODULE}.load_remote_job")
def test_check_remote_job_state_returns_terminal(
    mock_load_job: MagicMock,
) -> None:
    """Terminal state is returned immediately without SSH."""
    mock_load_job.return_value = _make_record(
        state="completed", model_path_remote="/out/model",
    )
    with patch(f"{_STATE_MODULE}.is_model_registered", return_value=True):
        result = check_remote_job_state(Path("/data"), "rj-test123")
    assert result == "completed"


@patch(f"{_STATE_MODULE}.update_remote_job_state")
@patch(f"{_STATE_MODULE}.load_cluster", return_value=_make_cluster())
@patch(f"{_STATE_MODULE}.load_remote_job")
def test_check_remote_job_state_queries_slurm(
    mock_load_job: MagicMock,
    mock_load_cluster: MagicMock,
    mock_update: MagicMock,
) -> None:
    """Running job queries Slurm and returns updated state."""
    mock_load_job.return_value = _make_record(state="running")
    mock_session = _make_session([
        ("COMPLETED|0:0|None\n", "", 0),  # sacct in sync_final_state
        ('{"status":"completed","model_path":"/out/m"}\n', "", 0),  # result err
        ('{"status":"completed","model_path":"/out/m"}\n', "", 0),  # discover
    ])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_session)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch(f"{_STATE_MODULE}.SshSession", return_value=ctx), \
         patch(f"{_STATE_MODULE}.auto_register_remote_model"):
        result = check_remote_job_state(Path("/data"), "rj-test123")
    assert result == "completed"
