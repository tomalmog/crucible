"""Tests for HttpApiRunner — mocks urllib to verify all HTTP operations."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleRemoteError
from core.job_types import JobSpec
from serve.http_api_runner import HttpApiRunner


def _make_spec() -> JobSpec:
    return JobSpec(
        job_type="sft",
        method_args={"dataset_name": "my-data"},
        backend="http-api",
        label="SFT remote",
        cluster_name="api-cluster",
    )


@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Create a minimal data root with an API-configured cluster."""
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    cluster_dict = {
        "name": "api-cluster",
        "host": "api-host",
        "user": "apiuser",
        "api_endpoint": "http://localhost:8080",
        "api_token": "test-token-123",
    }
    (clusters_dir / "api-cluster.json").write_text(json.dumps(cluster_dict))
    return tmp_path


class TestHttpApiRunnerKind:
    """Tests for HttpApiRunner.kind property."""

    def test_kind_is_http_api(self) -> None:
        """Runner should report its kind as 'http-api'."""
        runner = HttpApiRunner()
        assert runner.kind == "http-api"


class TestHttpApiRunnerSubmit:
    """Tests for HttpApiRunner.submit()."""

    @patch("serve.http_api_runner.urllib.request.urlopen")
    def test_submit_creates_local_record(
        self, mock_urlopen: MagicMock, tmp_data_root: Path,
    ) -> None:
        """Successful submit should create a local job record."""
        response_data = {
            "job_id": "remote-job-abc",
            "state": "running",
            "backend": "http-api",
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = HttpApiRunner()
        record = runner.submit(tmp_data_root, _make_spec())

        assert record.backend == "http-api"
        assert record.backend_job_id == "remote-job-abc"
        assert record.state == "running"


class TestHttpApiRunnerGetState:
    """Tests for HttpApiRunner.get_state()."""

    @patch("serve.http_api_runner.urllib.request.urlopen")
    def test_get_state_returns_state(
        self, mock_urlopen: MagicMock, tmp_data_root: Path,
    ) -> None:
        """get_state should return the state from the API response."""
        # Create a job first
        submit_response = {
            "job_id": "remote-123",
            "state": "running",
        }
        state_response = {
            "job_id": "remote-123",
            "state": "completed",
        }

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.side_effect = [
            json.dumps(submit_response).encode(),
            json.dumps(state_response).encode(),
        ]
        mock_urlopen.return_value = mock_resp

        runner = HttpApiRunner()
        record = runner.submit(tmp_data_root, _make_spec())

        state = runner.get_state(tmp_data_root, record.job_id)
        assert state == "completed"


class TestHttpApiRunnerGetLogs:
    """Tests for HttpApiRunner.get_logs()."""

    @patch("serve.http_api_runner.urllib.request.urlopen")
    def test_get_logs_returns_log_text(
        self, mock_urlopen: MagicMock, tmp_data_root: Path,
    ) -> None:
        """get_logs should return the log text from the API."""
        submit_response = {"job_id": "remote-123", "state": "running"}
        logs_response = {"logs": "epoch 1/3: loss=0.5\nepoch 2/3: loss=0.3"}

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.side_effect = [
            json.dumps(submit_response).encode(),
            json.dumps(logs_response).encode(),
        ]
        mock_urlopen.return_value = mock_resp

        runner = HttpApiRunner()
        record = runner.submit(tmp_data_root, _make_spec())

        logs = runner.get_logs(tmp_data_root, record.job_id)
        assert "loss=0.5" in logs


class TestHttpApiRunnerCancel:
    """Tests for HttpApiRunner.cancel()."""

    @patch("serve.http_api_runner.urllib.request.urlopen")
    def test_cancel_sets_cancelled_state(
        self, mock_urlopen: MagicMock, tmp_data_root: Path,
    ) -> None:
        """cancel should update local record to cancelled."""
        submit_response = {"job_id": "remote-123", "state": "running"}
        cancel_response = {"job_id": "remote-123", "state": "cancelled"}

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.side_effect = [
            json.dumps(submit_response).encode(),
            json.dumps(cancel_response).encode(),
        ]
        mock_urlopen.return_value = mock_resp

        runner = HttpApiRunner()
        record = runner.submit(tmp_data_root, _make_spec())

        updated = runner.cancel(tmp_data_root, record.job_id)
        assert updated.state == "cancelled"
