"""Unit tests for unified job store CRUD operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import CrucibleError
from core.job_types import JobRecord
from store.job_store import (
    delete_job,
    generate_job_id,
    list_jobs,
    load_job,
    now_iso,
    save_job,
    update_job,
    _record_to_dict,
    _dict_to_record,
)


def _make_record(job_id: str = "job-abc123def456") -> JobRecord:
    ts = now_iso()
    return JobRecord(
        job_id=job_id,
        backend="local",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        label="My-Model",
    )


# ── ID generation ───────────────────────────────────────────────────────


def test_generate_job_id_format() -> None:
    """Job IDs should start with job- and be unique."""
    jid = generate_job_id()
    assert jid.startswith("job-")
    assert len(jid) == 16  # "job-" + 12 hex chars
    assert jid != generate_job_id()


def test_now_iso_returns_iso_string() -> None:
    ts = now_iso()
    assert "T" in ts
    assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


# ── Save / Load round-trip ──────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    loaded = load_job(tmp_path, "job-abc123def456")
    assert loaded.job_id == record.job_id
    assert loaded.backend == "local"
    assert loaded.job_type == "sft"
    assert loaded.state == "running"
    assert loaded.label == "My-Model"
    assert loaded.is_sweep is False
    assert loaded.sweep_trial_count == 0
    assert loaded.model_path == ""
    assert loaded.error_message == ""
    assert loaded.progress_percent == 0.0


def test_save_creates_jobs_directory(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    assert (tmp_path / "jobs").is_dir()
    assert (tmp_path / "jobs" / "job-abc123def456.json").exists()


def test_save_writes_valid_json(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    raw = json.loads((tmp_path / "jobs" / "job-abc123def456.json").read_text())
    assert raw["job_id"] == "job-abc123def456"
    assert raw["backend"] == "local"
    assert raw["state"] == "running"


def test_load_missing_job_raises(tmp_path: Path) -> None:
    with pytest.raises(CrucibleError, match="not found"):
        load_job(tmp_path, "job-nonexistent12")


def test_load_invalid_json_file(tmp_path: Path) -> None:
    (tmp_path / "jobs").mkdir(parents=True)
    (tmp_path / "jobs" / "job-badjson00000.json").write_text("not json")
    with pytest.raises(Exception):
        load_job(tmp_path, "job-badjson00000")


# ── All fields round-trip ───────────────────────────────────────────────


def test_all_fields_round_trip(tmp_path: Path) -> None:
    ts = now_iso()
    record = JobRecord(
        job_id="job-fullfields00",
        backend="slurm",
        job_type="dpo-train",
        state="completed",
        created_at=ts,
        updated_at=ts,
        label="DPO-Run-1",
        backend_job_id="rj-legacy00001",
        backend_cluster="gpu-hpc",
        backend_output_dir="/scratch/crucible/run1",
        backend_log_path="/scratch/crucible/run1/log.out",
        model_path="/remote/model/path",
        model_path_local="/local/model/path",
        model_name="DPO-Model",
        error_message="",
        progress_percent=100.0,
        submit_phase="done",
        is_sweep=True,
        sweep_trial_count=5,
    )
    save_job(tmp_path, record)
    loaded = load_job(tmp_path, "job-fullfields00")
    assert loaded.backend == "slurm"
    assert loaded.job_type == "dpo-train"
    assert loaded.state == "completed"
    assert loaded.label == "DPO-Run-1"
    assert loaded.backend_job_id == "rj-legacy00001"
    assert loaded.backend_cluster == "gpu-hpc"
    assert loaded.backend_output_dir == "/scratch/crucible/run1"
    assert loaded.backend_log_path == "/scratch/crucible/run1/log.out"
    assert loaded.model_path == "/remote/model/path"
    assert loaded.model_path_local == "/local/model/path"
    assert loaded.model_name == "DPO-Model"
    assert loaded.progress_percent == 100.0
    assert loaded.submit_phase == "done"
    assert loaded.is_sweep is True
    assert loaded.sweep_trial_count == 5


# ── List ────────────────────────────────────────────────────────────────


def test_list_jobs_empty(tmp_path: Path) -> None:
    result = list_jobs(tmp_path)
    assert result == ()


def test_list_jobs_returns_all(tmp_path: Path) -> None:
    r1 = JobRecord(
        job_id="job-aaa000000001",
        backend="local",
        job_type="sft",
        state="completed",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    r2 = JobRecord(
        job_id="job-bbb000000002",
        backend="slurm",
        job_type="train",
        state="running",
        created_at="2026-02-01T00:00:00+00:00",
        updated_at="2026-02-01T00:00:00+00:00",
    )
    save_job(tmp_path, r1)
    save_job(tmp_path, r2)
    result = list_jobs(tmp_path)
    assert len(result) == 2
    # Newest first
    assert result[0].job_id == "job-bbb000000002"
    assert result[1].job_id == "job-aaa000000001"


def test_list_jobs_ignores_non_job_files(tmp_path: Path) -> None:
    """Non job-*.json files should be ignored."""
    (tmp_path / "jobs").mkdir(parents=True)
    (tmp_path / "jobs" / ".migrated_v1").write_text("migrated\n")
    (tmp_path / "jobs" / "random.json").write_text("{}")
    r = _make_record()
    save_job(tmp_path, r)
    result = list_jobs(tmp_path)
    assert len(result) == 1


# ── Update ──────────────────────────────────────────────────────────────


def test_update_job_changes_fields(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    updated = update_job(tmp_path, record.job_id, state="completed", model_path="/out/model")
    assert updated.state == "completed"
    assert updated.model_path == "/out/model"
    assert updated.updated_at != record.updated_at
    # Other fields unchanged
    assert updated.backend == "local"
    assert updated.label == "My-Model"


def test_update_nonexistent_raises(tmp_path: Path) -> None:
    with pytest.raises(CrucibleError, match="not found"):
        update_job(tmp_path, "job-doesnotexist", state="completed")


def test_update_persists_to_disk(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    update_job(tmp_path, record.job_id, state="failed", error_message="boom")
    reloaded = load_job(tmp_path, record.job_id)
    assert reloaded.state == "failed"
    assert reloaded.error_message == "boom"


# ── Delete ──────────────────────────────────────────────────────────────


def test_delete_job_removes_file(tmp_path: Path) -> None:
    record = _make_record()
    save_job(tmp_path, record)
    delete_job(tmp_path, record.job_id)
    assert not (tmp_path / "jobs" / "job-abc123def456.json").exists()


def test_delete_nonexistent_raises(tmp_path: Path) -> None:
    with pytest.raises(CrucibleError, match="not found"):
        delete_job(tmp_path, "job-nonexistent12")


# ── Serialization helpers ───────────────────────────────────────────────


def test_record_to_dict_has_all_keys() -> None:
    record = _make_record()
    d = _record_to_dict(record)
    expected_keys = {
        "job_id", "backend", "job_type", "state", "created_at", "updated_at",
        "label", "backend_job_id", "backend_cluster", "backend_output_dir",
        "backend_log_path", "model_path", "model_path_local", "model_name",
        "error_message", "progress_percent", "submit_phase", "is_sweep",
        "sweep_trial_count",
    }
    assert set(d.keys()) == expected_keys


def test_dict_to_record_handles_missing_keys() -> None:
    """Missing keys should default gracefully."""
    record = _dict_to_record({"job_id": "job-minimal00000", "state": "pending"})
    assert record.job_id == "job-minimal00000"
    assert record.backend == "local"
    assert record.job_type == ""
    assert record.state == "pending"
    assert record.label == ""
    assert record.is_sweep is False
    assert record.progress_percent == 0.0


def test_dict_to_record_handles_wrong_types() -> None:
    """Numeric strings and booleans should coerce correctly."""
    record = _dict_to_record({
        "job_id": "job-coerce000001",
        "backend": "slurm",
        "state": "running",
        "progress_percent": 50,  # int instead of float
        "is_sweep": 1,  # truthy int
        "sweep_trial_count": "3",  # string instead of int
    })
    assert record.progress_percent == 50.0
    assert record.is_sweep is True
    assert record.sweep_trial_count == 3
