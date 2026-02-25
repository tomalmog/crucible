"""End-to-end integration tests for cloud burst training workflow.

Tests cover cloud cost estimation, job submission, status polling,
result synchronization, and CLI entry points for the cloud command.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cli.main import main
from serve.cloud_burst import (
    CloudCostEstimate,
    CloudJobStatus,
    estimate_cloud_cost,
    poll_cloud_job,
    submit_cloud_job,
    sync_cloud_results,
)


def test_estimate_modal() -> None:
    """Modal A100 at 10 hours should cost 10 * 2.78 = 27.80."""
    est = estimate_cloud_cost(10.0, "modal", "a100")

    assert isinstance(est, CloudCostEstimate)
    assert est.provider == "modal"
    assert est.gpu_type == "a100"
    assert est.estimated_hours == 10.0
    assert est.cost_per_hour == pytest.approx(2.78)
    assert est.total_cost == pytest.approx(27.80)


def test_estimate_lambda() -> None:
    """Lambda A100 at 5 hours should cost 5 * 1.99 = 9.95."""
    est = estimate_cloud_cost(5.0, "lambda", "a100")

    assert isinstance(est, CloudCostEstimate)
    assert est.provider == "lambda"
    assert est.gpu_type == "a100"
    assert est.estimated_hours == 5.0
    assert est.cost_per_hour == pytest.approx(1.99)
    assert est.total_cost == pytest.approx(9.95)


def test_submit_returns_job_id() -> None:
    """Submitting a job should return a pending status with a non-empty job_id."""
    status = submit_cloud_job({}, "modal", "test-key")

    assert isinstance(status, CloudJobStatus)
    assert status.job_id
    assert len(status.job_id) > 0
    assert status.status == "pending"
    assert status.provider == "modal"


def test_poll_returns_running() -> None:
    """Polling a submitted job should return a valid status with non-negative progress."""
    submitted = submit_cloud_job({}, "runpod", "test-key")

    polled = poll_cloud_job(submitted.job_id, "runpod")

    assert isinstance(polled, CloudJobStatus)
    assert polled.job_id == submitted.job_id
    assert polled.status in ("running", "pending", "completed")
    assert polled.progress >= 0


def test_sync_creates_manifest(tmp_path: Path) -> None:
    """Syncing results should create a manifest.json inside cloud_results/<job_id>/."""
    submitted = submit_cloud_job({}, "modal", "test-key")

    output_path = sync_cloud_results(submitted.job_id, "modal", tmp_path)

    assert output_path
    manifest_path = Path(output_path) / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["job_id"] == submitted.job_id
    assert manifest["provider"] == "modal"
    assert manifest["synced"] is True

    # Also verify the path is under cloud_results/<job_id>
    expected_dir = tmp_path / "cloud_results" / submitted.job_id
    assert Path(output_path) == expected_dir


def test_cli_estimate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI cloud estimate should exit 0 and print total_cost."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "cloud", "estimate",
        "--hours", "10",
        "--provider", "modal",
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert re.search(r"(?i)total_cost", captured)


def test_cli_submit_and_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI submit should return a job_id; status should exit 0 for that job."""
    # Submit a job via CLI
    exit_code = main([
        "--data-root", str(tmp_path),
        "cloud", "submit",
        "--provider", "modal",
        "--api-key", "test-key",
    ])
    submit_output = capsys.readouterr().out
    assert exit_code == 0
    assert "job_id=" in submit_output

    # Extract job_id from output (format: job_id=<hash>)
    match = re.search(r"job_id=(\S+)", submit_output)
    assert match is not None
    job_id = match.group(1)

    # Check status via CLI
    exit_code = main([
        "--data-root", str(tmp_path),
        "cloud", "status",
        "--job-id", job_id,
        "--provider", "modal",
    ])
    status_output = capsys.readouterr().out

    assert exit_code == 0
    assert job_id in status_output
