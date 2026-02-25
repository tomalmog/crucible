"""Tests for cloud burst."""

from __future__ import annotations

from pathlib import Path

from serve.cloud_burst import (
    CloudCostEstimate,
    CloudJobStatus,
    estimate_cloud_cost,
    submit_cloud_job,
    poll_cloud_job,
    sync_cloud_results,
)


def test_estimate_cost_modal() -> None:
    """Estimate cost for Modal provider."""
    est = estimate_cloud_cost(2.0, "modal", "a100")
    assert est.provider == "modal"
    assert est.total_cost > 0
    assert est.estimated_hours == 2.0


def test_estimate_cost_lambda() -> None:
    """Lambda is generally cheapest."""
    modal = estimate_cloud_cost(1.0, "modal", "a100")
    lam = estimate_cloud_cost(1.0, "lambda", "a100")
    assert lam.total_cost <= modal.total_cost


def test_submit_job_returns_id() -> None:
    """Submitting a job returns a job ID."""
    status = submit_cloud_job({"method": "sft"}, "modal", "test-key")
    assert status.job_id
    assert status.provider == "modal"
    assert status.status == "pending"


def test_poll_job_status() -> None:
    """Polling returns running status."""
    status = poll_cloud_job("test-job-123", "modal")
    assert status.status == "running"
    assert status.progress > 0


def test_sync_results(tmp_path: Path) -> None:
    """Syncing creates output directory."""
    path = sync_cloud_results("test-job", "modal", tmp_path)
    assert Path(path).exists()
