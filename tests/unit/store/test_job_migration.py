"""Unit tests for job migration from remote-jobs to unified jobs."""

from __future__ import annotations

import json
from pathlib import Path

from core.job_types import JobRecord
from store.job_migration import (
    ensure_migrated,
    migrate_remote_jobs,
    _convert_remote_record,
    _needs_migration,
)
from store.job_store import list_jobs, load_job


def _write_legacy_record(data_root: Path, raw: dict[str, object]) -> None:
    """Write a legacy remote-jobs/*.json file."""
    remote_dir = data_root / "remote-jobs"
    remote_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(raw["job_id"])
    (remote_dir / f"{job_id}.json").write_text(json.dumps(raw, indent=2))


def _make_legacy_dict(job_id: str = "rj-abc123def456") -> dict[str, object]:
    return {
        "job_id": job_id,
        "slurm_job_id": "12345",
        "cluster_name": "test-hpc",
        "training_method": "sft",
        "state": "completed",
        "submitted_at": "2026-01-15T10:00:00+00:00",
        "updated_at": "2026-01-15T12:30:00+00:00",
        "remote_output_dir": "/scratch/crucible/rj-abc123def456",
        "remote_log_path": "/scratch/crucible/rj-abc123def456/slurm.out",
        "model_path_remote": "/scratch/crucible/rj-abc123def456/model.pt",
        "model_path_local": "",
        "model_name": "My-Model",
        "is_sweep": False,
        "sweep_array_size": 0,
        "submit_phase": "done",
    }


# ── _needs_migration ────────────────────────────────────────────────────


def test_needs_migration_no_marker(tmp_path: Path) -> None:
    assert _needs_migration(tmp_path) is True


def test_needs_migration_with_marker(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / ".migrated_v1").write_text("migrated\n")
    assert _needs_migration(tmp_path) is False


# ── _convert_remote_record ──────────────────────────────────────────────


def test_convert_remote_record_basic() -> None:
    raw = _make_legacy_dict()
    record = _convert_remote_record(raw)
    # Legacy ID becomes both job_id and backend_job_id
    assert record.job_id == "rj-abc123def456"
    assert record.backend_job_id == "rj-abc123def456"
    assert record.backend == "slurm"
    assert record.job_type == "sft"
    assert record.state == "completed"
    assert record.label == "My-Model"
    assert record.model_name == "My-Model"
    assert record.backend_cluster == "test-hpc"
    assert record.backend_output_dir == "/scratch/crucible/rj-abc123def456"
    assert record.backend_log_path == "/scratch/crucible/rj-abc123def456/slurm.out"
    assert record.model_path == "/scratch/crucible/rj-abc123def456/model.pt"
    assert record.created_at == "2026-01-15T10:00:00+00:00"
    assert record.updated_at == "2026-01-15T12:30:00+00:00"
    assert record.is_sweep is False
    assert record.sweep_trial_count == 0


def test_convert_remote_record_sweep() -> None:
    raw = _make_legacy_dict("rj-sweep0000001")
    raw["is_sweep"] = True
    raw["sweep_array_size"] = 10
    record = _convert_remote_record(raw)
    assert record.is_sweep is True
    assert record.sweep_trial_count == 10


def test_convert_remote_record_missing_fields() -> None:
    """Should handle minimal legacy records gracefully."""
    raw: dict[str, object] = {
        "job_id": "rj-minimal00001",
        "state": "running",
    }
    record = _convert_remote_record(raw)
    assert record.job_id == "rj-minimal00001"
    assert record.backend == "slurm"
    assert record.state == "running"
    assert record.job_type == ""
    assert record.label == ""


# ── migrate_remote_jobs ─────────────────────────────────────────────────


def test_migrate_no_remote_dir(tmp_path: Path) -> None:
    """No remote-jobs dir → writes marker, returns 0."""
    count = migrate_remote_jobs(tmp_path)
    assert count == 0
    assert (tmp_path / "jobs" / ".migrated_v1").exists()


def test_migrate_empty_remote_dir(tmp_path: Path) -> None:
    (tmp_path / "remote-jobs").mkdir()
    count = migrate_remote_jobs(tmp_path)
    assert count == 0
    assert (tmp_path / "jobs" / ".migrated_v1").exists()


def test_migrate_converts_records(tmp_path: Path) -> None:
    _write_legacy_record(tmp_path, _make_legacy_dict("rj-aaa000000001"))
    _write_legacy_record(tmp_path, _make_legacy_dict("rj-bbb000000002"))
    count = migrate_remote_jobs(tmp_path)
    assert count == 2
    # Check files exist in .crucible/jobs/
    assert (tmp_path / "jobs" / "rj-aaa000000001.json").exists()
    assert (tmp_path / "jobs" / "rj-bbb000000002.json").exists()
    # Check data is correct
    loaded = load_job(tmp_path, "rj-aaa000000001")
    assert loaded.backend == "slurm"
    assert loaded.backend_job_id == "rj-aaa000000001"


def test_migrate_preserves_legacy_files(tmp_path: Path) -> None:
    """Migration should not delete legacy files."""
    _write_legacy_record(tmp_path, _make_legacy_dict())
    migrate_remote_jobs(tmp_path)
    assert (tmp_path / "remote-jobs" / "rj-abc123def456.json").exists()


# ── ensure_migrated ─────────────────────────────────────────────────────


def test_ensure_migrated_runs_once(tmp_path: Path) -> None:
    _write_legacy_record(tmp_path, _make_legacy_dict())
    ensure_migrated(tmp_path)
    assert (tmp_path / "jobs" / "rj-abc123def456.json").exists()
    # Call again — should be a no-op (marker exists)
    ensure_migrated(tmp_path)
    # Still just 1 record
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1


def test_list_jobs_triggers_migration(tmp_path: Path) -> None:
    """list_jobs should auto-migrate on first call."""
    _write_legacy_record(tmp_path, _make_legacy_dict("rj-auto0000001"))
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].job_id == "rj-auto0000001"
    assert jobs[0].backend == "slurm"
