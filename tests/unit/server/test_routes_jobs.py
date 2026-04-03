"""Tests for job API routes — uses FastAPI TestClient with mocked backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.job_types import JobRecord


@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Create a minimal data root directory."""
    (tmp_path / "jobs").mkdir()
    return tmp_path


def _make_test_client(data_root: Path) -> object:
    """Create a FastAPI TestClient with job routes."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from server.routes_jobs import create_jobs_router

    app = fastapi.FastAPI()
    app.include_router(create_jobs_router(str(data_root)))
    return TestClient(app, raise_server_exceptions=False)


def _make_record() -> JobRecord:
    return JobRecord(
        job_id="job-test123abc",
        backend="local",
        job_type="sft",
        state="running",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        label="Test SFT",
    )


class TestListJobs:
    """Tests for GET /api/jobs."""

    def test_list_empty(self, tmp_data_root: Path) -> None:
        """Empty job store returns an empty list."""
        (tmp_data_root / "jobs" / ".migrated_v1").write_text("migrated\n")
        client = _make_test_client(tmp_data_root)
        response = client.get("/api/jobs")  # type: ignore[union-attr]
        assert response.status_code == 200  # type: ignore[union-attr]
        assert response.json() == []  # type: ignore[union-attr]


class TestGetJob:
    """Tests for GET /api/jobs/{job_id}."""

    def test_get_existing_job(self, tmp_data_root: Path) -> None:
        """Getting an existing job should return its data."""
        from store.job_store import save_job

        record = _make_record()
        save_job(tmp_data_root, record)

        client = _make_test_client(tmp_data_root)
        response = client.get(f"/api/jobs/{record.job_id}")  # type: ignore[union-attr]
        assert response.status_code == 200  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        assert data["job_id"] == record.job_id
        assert data["state"] == "running"

    def test_get_nonexistent_job_returns_error(self, tmp_data_root: Path) -> None:
        """Getting a non-existent job should return an error."""
        client = _make_test_client(tmp_data_root)
        response = client.get("/api/jobs/job-doesnotexist")  # type: ignore[union-attr]
        assert response.status_code == 500  # type: ignore[union-attr]


class TestCancelJob:
    """Tests for POST /api/jobs/{job_id}/cancel."""

    @patch("core.backend_registry.get_backend")
    def test_cancel_calls_backend(
        self, mock_get_backend: MagicMock, tmp_data_root: Path,
    ) -> None:
        """Cancel should delegate to the backend's cancel method."""
        from store.job_store import save_job

        record = _make_record()
        save_job(tmp_data_root, record)

        cancelled = JobRecord(
            job_id=record.job_id,
            backend="local",
            job_type="sft",
            state="cancelled",
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        mock_backend = MagicMock()
        mock_backend.cancel.return_value = cancelled
        mock_get_backend.return_value = mock_backend

        client = _make_test_client(tmp_data_root)
        response = client.post(f"/api/jobs/{record.job_id}/cancel")  # type: ignore[union-attr]
        assert response.status_code == 200  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        assert data["state"] == "cancelled"


class TestTokenAuth:
    """Tests for bearer token authentication."""

    def test_missing_token_rejected(self, tmp_data_root: Path) -> None:
        """Requests without token should be rejected when token is set."""
        os.environ["CRUCIBLE_API_TOKEN"] = "secret-token"
        try:
            (tmp_data_root / "jobs" / ".migrated_v1").write_text("migrated\n")
            client = _make_test_client(tmp_data_root)
            response = client.get("/api/jobs")  # type: ignore[union-attr]
            assert response.status_code == 401  # type: ignore[union-attr]
        finally:
            del os.environ["CRUCIBLE_API_TOKEN"]

    def test_valid_token_accepted(self, tmp_data_root: Path) -> None:
        """Requests with valid token should be accepted."""
        os.environ["CRUCIBLE_API_TOKEN"] = "secret-token"
        try:
            (tmp_data_root / "jobs" / ".migrated_v1").write_text("migrated\n")
            client = _make_test_client(tmp_data_root)
            response = client.get(  # type: ignore[union-attr]
                "/api/jobs",
                headers={"Authorization": "Bearer secret-token"},
            )
            assert response.status_code == 200  # type: ignore[union-attr]
        finally:
            del os.environ["CRUCIBLE_API_TOKEN"]
