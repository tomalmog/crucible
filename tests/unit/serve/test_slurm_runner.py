"""Unit tests for the SlurmRunner execution backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleError
from core.job_types import JobRecord, JobSpec, ResourceConfig
from core.slurm_types import RemoteJobRecord, SlurmResourceConfig
from serve.slurm_runner import SlurmRunner, _spec_to_slurm_resources
from store.job_store import save_job, now_iso


# ── _spec_to_slurm_resources ───────────────────────────────────────────


def test_spec_to_slurm_resources_defaults() -> None:
    spec = JobSpec(job_type="sft")
    result = _spec_to_slurm_resources(spec)
    assert isinstance(result, SlurmResourceConfig)
    assert result.nodes == 1
    assert result.gpus_per_node == 1
    assert result.memory == "32G"


def test_spec_to_slurm_resources_custom() -> None:
    resources = ResourceConfig(
        nodes=2,
        gpus_per_node=4,
        cpus_per_task=8,
        memory="64G",
        time_limit="12:00:00",
        partition="gpu",
        gpu_type="a100",
    )
    spec = JobSpec(job_type="sft", resources=resources)
    result = _spec_to_slurm_resources(spec)
    assert result.nodes == 2
    assert result.gpus_per_node == 4
    assert result.cpus_per_task == 8
    assert result.memory == "64G"
    assert result.time_limit == "12:00:00"
    assert result.partition == "gpu"
    assert result.gpu_type == "a100"


# ── SlurmRunner basic properties ───────────────────────────────────────


def test_slurm_runner_kind() -> None:
    runner = SlurmRunner()
    assert runner.kind == "slurm"


# ── _find_legacy_id ────────────────────────────────────────────────────


def test_find_legacy_id_from_backend_job_id(tmp_path: Path) -> None:
    runner = SlurmRunner()
    ts = now_iso()
    record = JobRecord(
        job_id="job-abc123def456",
        backend="slurm",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        backend_job_id="rj-legacyid0001",
    )
    assert runner._find_legacy_id(tmp_path, record) == "rj-legacyid0001"


def test_find_legacy_id_from_job_id(tmp_path: Path) -> None:
    """Migrated records use rj- as job_id directly."""
    runner = SlurmRunner()
    ts = now_iso()
    record = JobRecord(
        job_id="rj-migrated00001",
        backend="slurm",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        backend_job_id="rj-migrated00001",
    )
    assert runner._find_legacy_id(tmp_path, record) == "rj-migrated00001"


def test_find_legacy_id_no_rj_prefix_raises(tmp_path: Path) -> None:
    runner = SlurmRunner()
    ts = now_iso()
    record = JobRecord(
        job_id="job-nolegacy0001",
        backend="slurm",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        backend_job_id="not-an-rj-id",
    )
    with pytest.raises(CrucibleError, match="Cannot find legacy"):
        runner._find_legacy_id(tmp_path, record)


# ── _remote_to_unified ─────────────────────────────────────────────────


def test_remote_to_unified_basic() -> None:
    runner = SlurmRunner()
    remote_record = RemoteJobRecord(
        job_id="rj-test00000001",
        slurm_job_id="12345",
        cluster_name="gpu-hpc",
        training_method="sft",
        state="running",
        submitted_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T01:00:00+00:00",
        remote_output_dir="/scratch/run1",
        remote_log_path="/scratch/run1/slurm.out",
        model_path_remote="/scratch/run1/model.pt",
        model_name="Test-Model",
    )
    unified = runner._remote_to_unified(remote_record)
    assert unified.job_id.startswith("job-")
    assert unified.backend == "slurm"
    assert unified.job_type == "sft"
    assert unified.state == "running"
    assert unified.backend_job_id == "rj-test00000001"
    assert unified.backend_cluster == "gpu-hpc"
    assert unified.backend_output_dir == "/scratch/run1"
    assert unified.backend_log_path == "/scratch/run1/slurm.out"
    assert unified.model_path == "/scratch/run1/model.pt"
    assert unified.model_name == "Test-Model"
    assert unified.label == "SFT \u00b7 Test-Model"
    assert unified.created_at == "2026-01-01T00:00:00+00:00"


def test_remote_to_unified_sweep() -> None:
    runner = SlurmRunner()
    remote_record = RemoteJobRecord(
        job_id="rj-sweep0000001",
        slurm_job_id="99999",
        cluster_name="hpc",
        training_method="sft",
        state="pending",
        submitted_at="2026-03-01T00:00:00+00:00",
        updated_at="2026-03-01T00:00:00+00:00",
        remote_output_dir="/d",
        is_sweep=True,
        sweep_array_size=10,
    )
    unified = runner._remote_to_unified(remote_record)
    assert unified.is_sweep is True
    assert unified.sweep_trial_count == 10


def test_remote_to_unified_rejects_non_remote_record() -> None:
    runner = SlurmRunner()
    with pytest.raises(CrucibleError, match="Expected RemoteJobRecord"):
        runner._remote_to_unified({"not": "a record"})  # type: ignore[arg-type]


# ── cancel ──────────────────────────────────────────────────────────────


def test_cancel_updates_state(tmp_path: Path) -> None:
    from unittest.mock import patch

    runner = SlurmRunner()
    ts = now_iso()
    record = JobRecord(
        job_id="job-cancel000001",
        backend="slurm",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        backend_job_id="rj-cancel000001",
    )
    save_job(tmp_path, record)

    with patch("serve.remote_model_puller.cancel_remote_job"):
        updated = runner.cancel(tmp_path, "job-cancel000001")

    assert updated.state == "cancelled"


# ── get_state ───────────────────────────────────────────────────────────


def test_get_state_calls_remote_check(tmp_path: Path) -> None:
    from unittest.mock import patch

    runner = SlurmRunner()
    ts = now_iso()
    record = JobRecord(
        job_id="job-state0000001",
        backend="slurm",
        job_type="sft",
        state="running",
        created_at=ts,
        updated_at=ts,
        backend_job_id="rj-state0000001",
    )
    save_job(tmp_path, record)

    with patch("serve.remote_job_state.check_remote_job_state", return_value="completed"):
        state = runner.get_state(tmp_path, "job-state0000001")

    assert state == "completed"
