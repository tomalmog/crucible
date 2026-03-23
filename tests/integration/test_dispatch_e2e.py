"""End-to-end tests for the dispatch CLI command.

Tests the full chain: CLI → dispatch_command → backend → job_store.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.backend_registry import register_backend, _BACKENDS
from core.job_types import JobRecord, JobSpec
from serve.local_runner import LocalRunner
from store.job_store import list_jobs, load_job, now_iso
from cli.dispatch_command import run_dispatch_command


@dataclass
class FakeTrainingResult:
    model_path: Path | None = None


class FakeClient:
    """Minimal CrucibleClient stub."""
    def __init__(self, data_root: Path) -> None:
        self._config = MagicMock(data_root=data_root)


def _clear_and_register(tmp_path: Path) -> None:
    _BACKENDS.clear()
    register_backend("local", LocalRunner())


# ── dispatch → local → completed ────────────────────────────────────────


def test_dispatch_local_sft_completed(tmp_path: Path) -> None:
    """Full CLI dispatch: sft → local runner → completed → on disk."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=Path("/tmp/out/model.pt"))
    client = FakeClient(tmp_path)

    spec = {
        "job_type": "sft",
        "method_args": {"--epochs": "1", "--data-path": "/tmp/data.jsonl"},
        "backend": "local",
        "label": "Test-SFT",
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        exit_code = run_dispatch_command(client, args)

    assert exit_code == 0

    # Job was saved to disk
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.state == "completed"
    assert job.backend == "local"
    assert job.job_type == "sft"
    assert job.label == "Test-SFT"
    assert job.model_path == "/tmp/out/model.pt"


def test_dispatch_local_train_no_model(tmp_path: Path) -> None:
    """dispatch with eval → no model_path → completed."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=None)
    client = FakeClient(tmp_path)

    spec = {
        "job_type": "eval",
        "method_args": {"--benchmark": "mmlu"},
        "backend": "local",
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        exit_code = run_dispatch_command(client, args)

    assert exit_code == 0
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "completed"
    assert jobs[0].model_path == ""


def test_dispatch_local_failure_raises(tmp_path: Path) -> None:
    """dispatch that fails → record shows failed, command still propagates error."""
    _clear_and_register(tmp_path)

    client = FakeClient(tmp_path)

    spec = {
        "job_type": "sft",
        "method_args": {},
        "backend": "local",
        "label": "Fail-Test",
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", side_effect=RuntimeError("GPU error")),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        try:
            run_dispatch_command(client, args)
        except RuntimeError:
            pass

    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "failed"
    assert "GPU error" in jobs[0].error_message


def test_dispatch_with_resources(tmp_path: Path) -> None:
    """dispatch with resources → spec parsed correctly → local run ignores resources."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=Path("/out/model.pt"))
    client = FakeClient(tmp_path)

    spec = {
        "job_type": "sft",
        "method_args": {"--epochs": "2"},
        "backend": "local",
        "label": "Resource-Test",
        "resources": {
            "nodes": 4,
            "gpus_per_node": 8,
            "memory": "128G",
        },
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        exit_code = run_dispatch_command(client, args)

    assert exit_code == 0
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].state == "completed"


def test_dispatch_sweep_spec(tmp_path: Path) -> None:
    """dispatch with is_sweep + sweep_trials → parsed correctly."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=Path("/out/model.pt"))
    client = FakeClient(tmp_path)

    spec = {
        "job_type": "sft",
        "method_args": {"--epochs": "3"},
        "backend": "local",
        "label": "Sweep-Test",
        "is_sweep": True,
        "sweep_trials": [
            {"--lr": "1e-4"},
            {"--lr": "1e-5"},
            {"--lr": "1e-3"},
        ],
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        exit_code = run_dispatch_command(client, args)

    assert exit_code == 0
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].is_sweep is True


# ── dispatch → slurm (mocked) ──────────────────────────────────────────


def test_dispatch_slurm_routes_correctly(tmp_path: Path) -> None:
    """dispatch with backend=slurm → calls SlurmRunner.submit()."""
    from serve.slurm_runner import SlurmRunner
    from core.slurm_types import RemoteJobRecord

    _BACKENDS.clear()
    register_backend("local", LocalRunner())
    register_backend("slurm", SlurmRunner())

    client = FakeClient(tmp_path)

    fake_remote_record = RemoteJobRecord(
        job_id="rj-newsubmit001",
        slurm_job_id="88888",
        cluster_name="test-hpc",
        training_method="sft",
        state="pending",
        submitted_at=now_iso(),
        updated_at=now_iso(),
        remote_output_dir="/scratch/run",
        remote_log_path="/scratch/run/slurm.out",
        model_name="Slurm-Model",
    )

    spec = {
        "job_type": "sft",
        "method_args": {"--epochs": "5"},
        "backend": "slurm",
        "label": "Slurm-Test",
        "cluster_name": "test-hpc",
        "resources": {
            "partition": "gpu",
            "nodes": 2,
            "gpus_per_node": 4,
        },
    }
    args = argparse.Namespace(spec=json.dumps(spec))

    with patch(
        "serve.remote_job_submitter.submit_remote_job",
        return_value=fake_remote_record,
    ):
        exit_code = run_dispatch_command(client, args)

    assert exit_code == 0
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.backend == "slurm"
    assert job.state == "pending"
    assert job.backend_job_id == "rj-newsubmit001"
    assert job.backend_cluster == "test-hpc"
    assert job.model_name == "Slurm-Model"


# ── Job command pipeline ────────────────────────────────────────────────


def test_job_list_after_dispatch(tmp_path: Path) -> None:
    """dispatch → job list → shows the created job."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=Path("/out/m.pt"))
    client = FakeClient(tmp_path)

    spec = {"job_type": "train", "method_args": {}, "backend": "local", "label": "List-Test"}
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        run_dispatch_command(client, args)

    # Now test the job list command
    from cli.job_command import run_job_command
    list_args = argparse.Namespace(job_action="list")
    exit_code = run_job_command(client, list_args)
    assert exit_code == 0


def test_job_result_after_dispatch(tmp_path: Path) -> None:
    """dispatch → job result → returns CRUCIBLE_JSON."""
    _clear_and_register(tmp_path)

    fake_result = FakeTrainingResult(model_path=Path("/out/m.pt"))
    client = FakeClient(tmp_path)

    spec = {"job_type": "sft", "method_args": {}, "backend": "local"}
    args = argparse.Namespace(spec=json.dumps(spec))

    with (
        patch("serve.local_runner.dispatch_training", return_value=fake_result),
        patch("core.config.CrucibleConfig.from_env", return_value=MagicMock(data_root=tmp_path)),
        patch("store.dataset_sdk.CrucibleClient"),
    ):
        run_dispatch_command(client, args)

    jobs = list_jobs(tmp_path)
    job_id = jobs[0].job_id

    from cli.job_command import run_job_command
    result_args = argparse.Namespace(job_action="result", job_id=job_id)
    exit_code = run_job_command(client, result_args)
    assert exit_code == 0
