"""Unit tests for remote result.json reading."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from core.slurm_types import RemoteJobRecord
from serve.remote_result_reader import (
    extract_result_error,
    extract_result_model_path,
    read_remote_result,
)
from store.remote_job_store import now_iso


def _make_session(stdout: str, code: int) -> MagicMock:
    """Build a mock session with a single execute response."""
    session = MagicMock()
    session.execute = MagicMock(return_value=(stdout, "", code))
    return session


def _make_record() -> RemoteJobRecord:
    ts = now_iso()
    return RemoteJobRecord(
        job_id="rj-test123",
        slurm_job_id="12345",
        cluster_name="test-hpc",
        training_method="sft",
        state="completed",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir="/scratch/forge/rj-test123",
    )


def test_read_remote_result_returns_parsed_json() -> None:
    """Valid result.json is parsed into a dict."""
    data = {"status": "completed", "model_path": "/out/model"}
    session = _make_session(json.dumps(data), 0)
    assert read_remote_result(session, _make_record()) == data


def test_read_remote_result_returns_empty_on_missing() -> None:
    """Missing file returns empty dict."""
    session = _make_session("", 1)
    assert read_remote_result(session, _make_record()) == {}


def test_read_remote_result_returns_empty_on_invalid_json() -> None:
    """Invalid JSON returns empty dict."""
    session = _make_session("not json", 0)
    assert read_remote_result(session, _make_record()) == {}


def test_extract_result_error_returns_error_string() -> None:
    """Failed result.json yields the error message."""
    data = {"status": "failed", "error": "OOM killed"}
    session = _make_session(json.dumps(data), 0)
    assert extract_result_error(session, _make_record()) == "OOM killed"


def test_extract_result_error_returns_empty_on_success() -> None:
    """Completed result.json yields empty string."""
    data = {"status": "completed", "model_path": "/out"}
    session = _make_session(json.dumps(data), 0)
    assert extract_result_error(session, _make_record()) == ""


def test_extract_result_model_path_returns_path() -> None:
    """result.json with model_path yields the path."""
    data = {"status": "completed", "model_path": "/out/model"}
    session = _make_session(json.dumps(data), 0)
    assert extract_result_model_path(session, _make_record()) == "/out/model"


def test_extract_result_model_path_returns_empty_on_missing() -> None:
    """Missing file yields empty string."""
    session = _make_session("", 1)
    assert extract_result_model_path(session, _make_record()) == ""
