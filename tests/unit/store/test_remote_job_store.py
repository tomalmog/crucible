"""Unit tests for remote job store CRUD operations."""

from __future__ import annotations

import pytest

from core.errors import CrucibleRemoteError
from core.slurm_types import RemoteJobRecord
from store.remote_job_store import (
    generate_job_id,
    list_remote_jobs,
    load_remote_job,
    now_iso,
    save_remote_job,
    update_remote_job_state,
)


def _make_record(job_id: str = "rj-abc123def456") -> RemoteJobRecord:
    ts = now_iso()
    return RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="12345",
        cluster_name="test-hpc",
        training_method="sft",
        state="running",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir="/scratch/crucible/rj-abc123def456",
        remote_log_path="/scratch/crucible/rj-abc123def456/slurm-12345.out",
    )


def test_generate_job_id_format() -> None:
    """Job IDs should start with rj- and be unique."""
    jid = generate_job_id()
    assert jid.startswith("rj-")
    assert len(jid) == 15  # "rj-" + 12 hex chars
    assert jid != generate_job_id()


def test_save_and_load_remote_job(tmp_path: object) -> None:
    """save_remote_job then load_remote_job should round-trip."""
    record = _make_record()
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    loaded = load_remote_job(tmp_path, "rj-abc123def456")  # type: ignore[arg-type]
    assert loaded.job_id == record.job_id
    assert loaded.slurm_job_id == "12345"
    assert loaded.cluster_name == "test-hpc"
    assert loaded.training_method == "sft"
    assert loaded.state == "running"
    assert loaded.remote_output_dir == record.remote_output_dir
    assert loaded.remote_log_path == record.remote_log_path
    assert loaded.model_path_remote == ""
    assert loaded.model_path_local == ""
    assert loaded.is_sweep is False
    assert loaded.sweep_array_size == 0


def test_load_missing_job_raises(tmp_path: object) -> None:
    """load_remote_job should raise CrucibleRemoteError for nonexistent job."""
    with pytest.raises(CrucibleRemoteError, match="not found"):
        load_remote_job(tmp_path, "rj-nonexistent12")  # type: ignore[arg-type]


def test_list_remote_jobs_empty(tmp_path: object) -> None:
    """list_remote_jobs should return empty tuple with no jobs."""
    result = list_remote_jobs(tmp_path)  # type: ignore[arg-type]
    assert result == ()


def test_list_remote_jobs_sorted_by_time(tmp_path: object) -> None:
    """list_remote_jobs should return newest first."""
    r1 = RemoteJobRecord(
        job_id="rj-aaa000000001",
        slurm_job_id="1",
        cluster_name="c",
        training_method="sft",
        state="completed",
        submitted_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        remote_output_dir="/d",
    )
    r2 = RemoteJobRecord(
        job_id="rj-bbb000000002",
        slurm_job_id="2",
        cluster_name="c",
        training_method="sft",
        state="running",
        submitted_at="2026-02-01T00:00:00+00:00",
        updated_at="2026-02-01T00:00:00+00:00",
        remote_output_dir="/d",
    )
    save_remote_job(tmp_path, r1)  # type: ignore[arg-type]
    save_remote_job(tmp_path, r2)  # type: ignore[arg-type]
    result = list_remote_jobs(tmp_path)  # type: ignore[arg-type]
    assert len(result) == 2
    assert result[0].job_id == "rj-bbb000000002"  # newer first
    assert result[1].job_id == "rj-aaa000000001"


def test_update_remote_job_state(tmp_path: object) -> None:
    """update_remote_job_state should change state and update timestamp."""
    record = _make_record()
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    updated = update_remote_job_state(tmp_path, record.job_id, "completed")  # type: ignore[arg-type]
    assert updated.state == "completed"
    assert updated.updated_at != record.updated_at


def test_update_remote_job_extra_fields(tmp_path: object) -> None:
    """update_remote_job_state should accept extra keyword fields."""
    record = _make_record()
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    updated = update_remote_job_state(
        tmp_path,  # type: ignore[arg-type]
        record.job_id,
        "completed",
        model_path_remote="/remote/model",
        model_path_local="/local/model",
    )
    assert updated.model_path_remote == "/remote/model"
    assert updated.model_path_local == "/local/model"


def test_model_name_round_trip(tmp_path: object) -> None:
    """model_name should persist through save/load cycle."""
    record = RemoteJobRecord(
        job_id="rj-named0000001",
        slurm_job_id="777",
        cluster_name="c",
        training_method="train",
        state="running",
        submitted_at=now_iso(),
        updated_at=now_iso(),
        remote_output_dir="/d",
        model_name="My-Transformer-v2",
    )
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    loaded = load_remote_job(tmp_path, "rj-named0000001")  # type: ignore[arg-type]
    assert loaded.model_name == "My-Transformer-v2"


def test_model_name_defaults_empty(tmp_path: object) -> None:
    """model_name should default to empty string for old records."""
    record = _make_record()
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    loaded = load_remote_job(tmp_path, record.job_id)  # type: ignore[arg-type]
    assert loaded.model_name == ""


def test_sweep_fields_round_trip(tmp_path: object) -> None:
    """Sweep-specific fields should persist correctly."""
    record = RemoteJobRecord(
        job_id="rj-sweep0000001",
        slurm_job_id="999",
        cluster_name="c",
        training_method="sft",
        state="running",
        submitted_at=now_iso(),
        updated_at=now_iso(),
        remote_output_dir="/d",
        is_sweep=True,
        sweep_array_size=20,
    )
    save_remote_job(tmp_path, record)  # type: ignore[arg-type]
    loaded = load_remote_job(tmp_path, "rj-sweep0000001")  # type: ignore[arg-type]
    assert loaded.is_sweep is True
    assert loaded.sweep_array_size == 20
