"""Unit tests for remote Slurm job submission orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig, RemoteJobRecord, SlurmResourceConfig
from serve.remote_job_submitter import submit_remote_job

_MODULE = "serve.remote_job_submitter"

_JOB_ID = "rj-test123"
_TS = "2026-01-01T00:00:00Z"
_CLUSTER_NAME = "test-cluster"
_WORKSPACE = "/scratch/forge"
_WORKDIR = f"{_WORKSPACE}/{_JOB_ID}"


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(
        name=_CLUSTER_NAME,
        host="test.example.com",
        user="testuser",
        remote_workspace=_WORKSPACE,
    )


def _make_resources() -> SlurmResourceConfig:
    return SlurmResourceConfig(
        nodes=1,
        gpus_per_node=1,
        gpu_type="",
        cpus_per_task=4,
        memory="32G",
        time_limit="24:00:00",
        partition="",
        extra_sbatch=(),
    )


def _make_result_record(state: str = "running") -> RemoteJobRecord:
    return RemoteJobRecord(
        job_id=_JOB_ID,
        slurm_job_id="99999",
        cluster_name=_CLUSTER_NAME,
        training_method="sft",
        state=state,
        submitted_at=_TS,
        updated_at=_TS,
        remote_output_dir=_WORKDIR,
    )


def _make_session_mock() -> MagicMock:
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.execute.return_value = ("", "", 0)
    return session


@pytest.fixture()
def _patch_all() -> dict[str, MagicMock]:
    """Patch every external dependency of submit_remote_job."""
    cluster = _make_cluster()
    session = _make_session_mock()
    result_record = _make_result_record()

    patches = {
        "load_cluster": patch(f"{_MODULE}.load_cluster", return_value=cluster),
        "generate_job_id": patch(f"{_MODULE}.generate_job_id", return_value=_JOB_ID),
        "now_iso": patch(f"{_MODULE}.now_iso", return_value=_TS),
        "build_agent_tarball": patch(
            f"{_MODULE}.build_agent_tarball",
            return_value=Path("/tmp/fake.tar.gz"),
        ),
        "save_remote_job": patch(f"{_MODULE}.save_remote_job"),
        "update_remote_job_state": patch(
            f"{_MODULE}.update_remote_job_state",
            return_value=result_record,
        ),
        "ensure_remote_env": patch(f"{_MODULE}.ensure_remote_env"),
        "SshSession": patch(f"{_MODULE}.SshSession", return_value=session),
        "_upload_bundle": patch(f"{_MODULE}._upload_bundle"),
        "_upload_config": patch(f"{_MODULE}._upload_config"),
        "_upload_script": patch(f"{_MODULE}._upload_script"),
        "_generate_script": patch(
            f"{_MODULE}._generate_script",
            return_value="#!/bin/bash\n",
        ),
        "_submit_sbatch": patch(f"{_MODULE}._submit_sbatch", return_value="99999"),
    }

    mocks: dict[str, MagicMock] = {}
    for key, patcher in patches.items():
        mocks[key] = patcher.start()

    yield mocks

    for patcher in patches.values():
        patcher.stop()


def _call_submit(
    method_args: dict[str, object] | None = None,
) -> RemoteJobRecord:
    return submit_remote_job(
        data_root=Path("/tmp/forge"),
        cluster_name=_CLUSTER_NAME,
        training_method="sft",
        method_args=method_args or {},
        resources=_make_resources(),
    )


def test_submit_remote_job_saves_initial_record(
    _patch_all: dict[str, MagicMock],
) -> None:
    """save_remote_job is called with a record in 'submitting' state."""
    _call_submit()
    saved_record = _patch_all["save_remote_job"].call_args.args[1]
    assert saved_record.state == "submitting"


def test_submit_remote_job_returns_record(
    _patch_all: dict[str, MagicMock],
) -> None:
    """Return value is the record produced by update_remote_job_state."""
    result = _call_submit()
    assert result == _make_result_record()


def test_submit_remote_job_calls_ensure_env(
    _patch_all: dict[str, MagicMock],
) -> None:
    """ensure_remote_env is invoked once during submission."""
    _call_submit()
    assert _patch_all["ensure_remote_env"].call_count == 1


def test_submit_remote_job_uploads_bundle(
    _patch_all: dict[str, MagicMock],
) -> None:
    """_upload_bundle is invoked once during submission."""
    _call_submit()
    assert _patch_all["_upload_bundle"].call_count == 1


def test_submit_remote_job_submits_sbatch(
    _patch_all: dict[str, MagicMock],
) -> None:
    """_submit_sbatch is invoked once during submission."""
    _call_submit()
    assert _patch_all["_submit_sbatch"].call_count == 1


def test_submit_remote_job_verifies_dataset(
    _patch_all: dict[str, MagicMock],
) -> None:
    """No error is raised when dataset directory exists on the remote."""
    session = _patch_all["SshSession"].return_value.__enter__.return_value
    session.execute.return_value = ("", "", 0)
    _call_submit(method_args={"dataset_name": "test-ds"})
    assert session.execute.call_count >= 1


def test_submit_remote_job_raises_on_missing_dataset(
    _patch_all: dict[str, MagicMock],
) -> None:
    """ForgeRemoteError is raised when the dataset directory is absent."""
    session = _patch_all["SshSession"].return_value.__enter__.return_value
    session.execute.return_value = ("", "", 1)
    with pytest.raises(ForgeRemoteError, match="not found on cluster"):
        _call_submit(method_args={"dataset_name": "test-ds"})


def test_submit_remote_job_marks_failed_on_error(
    _patch_all: dict[str, MagicMock],
) -> None:
    """Job state is set to 'failed' when _submit_sbatch raises."""
    _patch_all["_submit_sbatch"].side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        _call_submit()
    fail_call = _patch_all["update_remote_job_state"].call_args_list[-1]
    assert fail_call.args[2] == "failed"
