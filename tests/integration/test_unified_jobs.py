"""End-to-end integration tests for the unified job system.

Tests the full pipeline: dispatch → backend → job store → list/sync/result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.backend_registry import register_backend, _BACKENDS
from core.job_types import BackendKind, JobRecord, JobSpec, JobState, ResourceConfig
from store.job_store import (
    delete_job,
    generate_job_id,
    list_jobs,
    load_job,
    now_iso,
    save_job,
    update_job,
)
from store.job_migration import ensure_migrated, migrate_remote_jobs


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    _BACKENDS.clear()


# ── Full local pipeline: submit → record → list → result ──────────────


@dataclass
class FakeTrainingResult:
    model_path: Path | None = None


def test_local_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full local pipeline: register → submit → list → get_state → result."""
    from serve.local_runner import LocalRunner

    runner = LocalRunner()
    register_backend("local", runner)

    spec = JobSpec(
        job_type="sft",
        method_args={"--epochs": "3", "--data-path": "/tmp/data.jsonl"},
        backend="local",
        label="My-SFT-Model",
    )

    fake_result = FakeTrainingResult(model_path=Path("/tmp/output/model.pt"))

    fake_config = MagicMock(data_root=tmp_path)
    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        record = runner.submit(tmp_path, spec)

    # 1. Record exists on disk
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].job_id == record.job_id
    assert jobs[0].state == "completed"
    assert jobs[0].backend == "local"
    assert jobs[0].job_type == "sft"
    assert jobs[0].model_path == "/tmp/output/model.pt"
    assert jobs[0].label == "My-SFT-Model"

    # 2. get_state reflects completed
    state = runner.get_state(tmp_path, record.job_id)
    assert state == "completed"

    # 3. get_result has model_path
    result = runner.get_result(tmp_path, record.job_id)
    assert result["model_path"] == "/tmp/output/model.pt"
    assert result["state"] == "completed"

    # 4. Can be deleted
    delete_job(tmp_path, record.job_id)
    assert list_jobs(tmp_path) == ()


def test_local_failed_pipeline(tmp_path: Path) -> None:
    """Failed local job: submit errors → record shows failed."""
    from serve.local_runner import LocalRunner

    runner = LocalRunner()
    spec = JobSpec(
        job_type="sft",
        method_args={"--data-path": "/nonexistent"},
        backend="local",
        label="Will-Fail",
    )

    fake_config = MagicMock(data_root=tmp_path)
    with (
        patch("serve.local_runner.dispatch_training", side_effect=ValueError("Data not found")),
        patch("core.config.CrucibleConfig.from_env", return_value=fake_config),
        patch("store.dataset_sdk.CrucibleClient"),
        pytest.raises(ValueError),
    ):
        runner.submit(tmp_path, spec)

    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "failed"
    assert "Data not found" in jobs[0].error_message
    assert jobs[0].label == "Will-Fail"


# ── Migration pipeline: legacy → unified → list/query ──────────────────


def _write_legacy(data_root: Path, job_id: str, **overrides: object) -> None:
    """Write a legacy remote-jobs file."""
    raw: dict[str, object] = {
        "job_id": job_id,
        "slurm_job_id": "99999",
        "cluster_name": "test-hpc",
        "training_method": "sft",
        "state": "completed",
        "submitted_at": "2026-01-15T10:00:00+00:00",
        "updated_at": "2026-01-15T12:30:00+00:00",
        "remote_output_dir": f"/scratch/crucible/{job_id}",
        "remote_log_path": f"/scratch/crucible/{job_id}/slurm.out",
        "model_path_remote": f"/scratch/crucible/{job_id}/model.pt",
        "model_path_local": "",
        "model_name": "Test-Model",
        "is_sweep": False,
        "sweep_array_size": 0,
        "submit_phase": "done",
    }
    raw.update(overrides)
    remote_dir = data_root / "remote-jobs"
    remote_dir.mkdir(parents=True, exist_ok=True)
    (remote_dir / f"{job_id}.json").write_text(json.dumps(raw, indent=2))


def test_migration_end_to_end(tmp_path: Path) -> None:
    """Migration: create legacy records → list_jobs → see migrated records."""
    _write_legacy(tmp_path, "rj-aaa000000001")
    _write_legacy(tmp_path, "rj-bbb000000002", state="running")
    _write_legacy(tmp_path, "rj-sweep0000001", is_sweep=True, sweep_array_size=5)

    # list_jobs triggers auto-migration
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 3

    # All should have backend=slurm
    assert all(j.backend == "slurm" for j in jobs)

    # Legacy IDs preserved
    ids = {j.job_id for j in jobs}
    assert "rj-aaa000000001" in ids
    assert "rj-bbb000000002" in ids
    assert "rj-sweep0000001" in ids

    # Sweep detected
    sweep_job = next(j for j in jobs if j.job_id == "rj-sweep0000001")
    assert sweep_job.is_sweep is True
    assert sweep_job.sweep_trial_count == 5

    # Running job preserved
    running_job = next(j for j in jobs if j.job_id == "rj-bbb000000002")
    assert running_job.state == "running"

    # Legacy files untouched
    assert (tmp_path / "remote-jobs" / "rj-aaa000000001.json").exists()

    # Second list_jobs should not re-migrate
    jobs2 = list_jobs(tmp_path)
    assert len(jobs2) == 3


def test_migration_then_new_job(tmp_path: Path) -> None:
    """After migration, new local jobs coexist with migrated records."""
    _write_legacy(tmp_path, "rj-legacy000001")

    # Trigger migration
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1

    # Add a new local job
    ts = now_iso()
    new_record = JobRecord(
        job_id=generate_job_id(),
        backend="local",
        job_type="train",
        state="completed",
        created_at=ts,
        updated_at=ts,
        label="New-Local",
    )
    save_job(tmp_path, new_record)

    jobs = list_jobs(tmp_path)
    assert len(jobs) == 2
    backends = {j.backend for j in jobs}
    assert backends == {"local", "slurm"}


# ── Dispatch command JSON spec parsing ──────────────────────────────────


def test_dispatch_spec_parsing_full() -> None:
    """Verify dispatch command parses all JSON fields correctly."""
    from core.job_types import JobSpec, ResourceConfig

    raw = {
        "job_type": "dpo-train",
        "method_args": {"--epochs": "5", "--lr": "1e-5"},
        "backend": "slurm",
        "label": "DPO-Experiment",
        "cluster_name": "gpu-hpc",
        "resources": {
            "nodes": 2,
            "gpus_per_node": 4,
            "cpus_per_task": 8,
            "memory": "64G",
            "time_limit": "12:00:00",
            "partition": "a100",
            "gpu_type": "a100",
        },
        "is_sweep": True,
        "sweep_trials": [
            {"--lr": "1e-4"},
            {"--lr": "1e-5"},
        ],
    }

    # This mirrors what run_dispatch_command does internally
    resources = None
    if "resources" in raw and raw["resources"]:
        r = raw["resources"]
        resources = ResourceConfig(
            nodes=int(r.get("nodes", 1)),
            gpus_per_node=int(r.get("gpus_per_node", 1)),
            cpus_per_task=int(r.get("cpus_per_task", 4)),
            memory=str(r.get("memory", "32G")),
            time_limit=str(r.get("time_limit", "04:00:00")),
            partition=str(r.get("partition", "")),
            gpu_type=str(r.get("gpu_type", "")),
        )

    sweep_trials = tuple(dict(t) for t in raw.get("sweep_trials", []))

    spec = JobSpec(
        job_type=str(raw["job_type"]),
        method_args=dict(raw.get("method_args", {})),
        backend=str(raw.get("backend", "local")),  # type: ignore[arg-type]
        label=str(raw.get("label", "")),
        cluster_name=str(raw.get("cluster_name", "")),
        resources=resources,
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_trials=sweep_trials,
    )

    assert spec.job_type == "dpo-train"
    assert spec.backend == "slurm"
    assert spec.label == "DPO-Experiment"
    assert spec.cluster_name == "gpu-hpc"
    assert spec.is_sweep is True
    assert len(spec.sweep_trials) == 2
    assert spec.resources is not None
    assert spec.resources.nodes == 2
    assert spec.resources.gpus_per_node == 4
    assert spec.resources.memory == "64G"


# ── Rust task store JSON format verification ────────────────────────────


def test_rust_job_record_json_format(tmp_path: Path) -> None:
    """Verify the JSON format Rust writes is readable by Python job store."""
    # Simulate what Rust write_job_record() produces
    rust_json = {
        "job_id": "crucible-task-1",
        "backend": "local",
        "job_type": "sft",
        "state": "running",
        "created_at": "2026-03-23T10:00:00Z",
        "updated_at": "2026-03-23T10:00:00Z",
        "label": "",
        "backend_job_id": "",
        "backend_cluster": "",
        "backend_output_dir": "",
        "backend_log_path": "",
        "model_path": "",
        "model_path_local": "",
        "model_name": "",
        "error_message": "",
        "progress_percent": 0.0,
        "submit_phase": "",
        "is_sweep": False,
        "sweep_trial_count": 0,
    }
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "crucible-task-1.json").write_text(json.dumps(rust_json, indent=2))

    # Python can read it (even though ID doesn't start with "job-")
    record = load_job(tmp_path, "crucible-task-1")
    assert record.job_id == "crucible-task-1"
    assert record.backend == "local"
    assert record.state == "running"


def test_rust_job_record_after_update(tmp_path: Path) -> None:
    """Verify Python can read Rust's updated record."""
    rust_json = {
        "job_id": "crucible-task-2",
        "backend": "local",
        "job_type": "train",
        "state": "completed",
        "created_at": "2026-03-23T10:00:00Z",
        "updated_at": "2026-03-23T10:30:00Z",
        "label": "",
        "backend_job_id": "",
        "backend_cluster": "",
        "backend_output_dir": "",
        "backend_log_path": "",
        "model_path": "/output/model.pt",
        "model_path_local": "/output/model.pt",
        "model_name": "",
        "error_message": "",
        "progress_percent": 100.0,
        "submit_phase": "",
        "is_sweep": False,
        "sweep_trial_count": 0,
    }
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "crucible-task-2.json").write_text(json.dumps(rust_json, indent=2))

    record = load_job(tmp_path, "crucible-task-2")
    assert record.state == "completed"
    assert record.model_path == "/output/model.pt"
    assert record.progress_percent == 100.0


def test_rust_and_python_records_coexist(tmp_path: Path) -> None:
    """Rust-generated (crucible-task-*) and Python-generated (job-*) records
    should both be listed by list_jobs."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)

    # Write marker so migration doesn't interfere
    (jobs_dir / ".migrated_v1").write_text("done")

    # Rust-style record (crucible-task-*)
    rust_record = {
        "job_id": "crucible-task-5",
        "backend": "local",
        "job_type": "sft",
        "state": "completed",
        "created_at": "2026-03-23T10:00:00Z",
        "updated_at": "2026-03-23T10:30:00Z",
        "label": "",
        "backend_job_id": "",
        "backend_cluster": "",
        "backend_output_dir": "",
        "backend_log_path": "",
        "model_path": "/output/model.pt",
        "model_path_local": "/output/model.pt",
        "model_name": "",
        "error_message": "",
        "progress_percent": 100.0,
        "submit_phase": "",
        "is_sweep": False,
        "sweep_trial_count": 0,
    }
    (jobs_dir / "crucible-task-5.json").write_text(json.dumps(rust_record))

    # Python-style record (job-*)
    ts = now_iso()
    save_job(tmp_path, JobRecord(
        job_id="job-python000001",
        backend="local",
        job_type="train",
        state="running",
        created_at=ts,
        updated_at=ts,
    ))

    jobs = list_jobs(tmp_path)
    # Both crucible-task-* and job-* should be listed
    assert len(jobs) == 2
    ids = {j.job_id for j in jobs}
    assert "crucible-task-5" in ids
    assert "job-python000001" in ids


# ── Edge cases ──────────────────────────────────────────────────────────


def test_update_then_list_shows_updated(tmp_path: Path) -> None:
    ts = now_iso()
    record = JobRecord(
        job_id="job-update000001",
        backend="local",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
    )
    save_job(tmp_path, record)

    # Update to completed
    update_job(tmp_path, "job-update000001", state="completed", model_path="/out/m.pt")

    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "completed"
    assert jobs[0].model_path == "/out/m.pt"


def test_delete_then_list_is_empty(tmp_path: Path) -> None:
    ts = now_iso()
    record = JobRecord(
        job_id="job-delete000001",
        backend="local",
        job_type="sft",
        state="completed",
        created_at=ts,
        updated_at=ts,
    )
    save_job(tmp_path, record)
    delete_job(tmp_path, "job-delete000001")
    assert list_jobs(tmp_path) == ()


def test_save_overwrite_existing(tmp_path: Path) -> None:
    """Saving with same job_id should overwrite."""
    ts = now_iso()
    r1 = JobRecord(
        job_id="job-overwrite0001",
        backend="local",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
    )
    save_job(tmp_path, r1)

    r2 = JobRecord(
        job_id="job-overwrite0001",
        backend="local",
        job_type="sft",
        state="completed",
        created_at=ts,
        updated_at=ts,
        model_path="/out/m.pt",
    )
    save_job(tmp_path, r2)

    loaded = load_job(tmp_path, "job-overwrite0001")
    assert loaded.state == "completed"
    assert loaded.model_path == "/out/m.pt"
