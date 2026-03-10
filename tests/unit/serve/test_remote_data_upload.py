"""Unit tests for remote data upload helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.remote_data_upload import (
    _generate_script,
    _submit_sbatch,
    _upload_bundle,
    _upload_config,
    _upload_script,
)


def _make_session() -> MagicMock:
    """Build a mock SshSession."""
    return MagicMock()


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(
        name="test",
        host="test.example.com",
        user="testuser",
        remote_workspace="/scratch/crucible",
    )


def _make_resources(nodes: int = 1) -> SlurmResourceConfig:
    return SlurmResourceConfig(
        nodes=nodes,
        gpus_per_node=1,
        gpu_type="",
        cpus_per_task=4,
        memory="32G",
        time_limit="24:00:00",
        partition="",
        extra_sbatch=(),
    )


def test_upload_bundle_calls_upload() -> None:
    """session.upload is called with the tarball and workdir paths."""
    session = _make_session()
    tarball = Path("/tmp/bundle.tar.gz")
    _upload_bundle(session, tarball, "/remote/work")
    session.upload.assert_called_once_with(tarball, "/remote/work/crucible-agent.tar.gz")


def test_upload_config_calls_upload_text() -> None:
    """session.upload_text receives JSON content and correct remote path."""
    session = _make_session()
    config = {"lr": 0.001, "epochs": 3}
    _upload_config(session, config, "/remote/work")
    session.upload_text.assert_called_once()
    content, remote = session.upload_text.call_args.args
    assert json.loads(content) == config
    assert remote == "/remote/work/training_config.json"


def test_upload_script_calls_upload_text() -> None:
    """session.upload_text receives script content and job.sh path."""
    session = _make_session()
    _upload_script(session, "#!/bin/bash\necho hello", "/remote/work")
    session.upload_text.assert_called_once()
    content, remote = session.upload_text.call_args.args
    assert content == "#!/bin/bash\necho hello"
    assert remote == "/remote/work/job.sh"


def test_submit_sbatch_parses_job_id() -> None:
    """Parses Slurm job ID from standard sbatch output."""
    session = _make_session()
    session.execute.return_value = ("Submitted batch job 12345\n", "", 0)
    result = _submit_sbatch(session, "/remote/work")
    assert result == "12345"


def test_submit_sbatch_raises_on_nonzero_exit() -> None:
    """Non-zero exit code raises CrucibleRemoteError."""
    session = _make_session()
    session.execute.return_value = ("", "sbatch: error: invalid partition", 1)
    with pytest.raises(CrucibleRemoteError, match="sbatch failed"):
        _submit_sbatch(session, "/remote/work")


def test_submit_sbatch_raises_on_unexpected_output() -> None:
    """Stdout with fewer than 4 words raises CrucibleRemoteError."""
    session = _make_session()
    session.execute.return_value = ("Error\n", "", 0)
    with pytest.raises(CrucibleRemoteError, match="Unexpected sbatch output"):
        _submit_sbatch(session, "/remote/work")


@patch("serve.remote_data_upload.generate_single_node_script", return_value="single")
def test_generate_script_single_node(mock_single: MagicMock) -> None:
    """Single-node resources dispatches to generate_single_node_script."""
    cluster = _make_cluster()
    resources = _make_resources(nodes=1)
    _generate_script(cluster, resources, "job-1", "sft")
    mock_single.assert_called_once_with(cluster, resources, "job-1", "sft")


@patch("serve.remote_data_upload.generate_multi_node_script", return_value="multi")
def test_generate_script_multi_node(mock_multi: MagicMock) -> None:
    """Multi-node resources dispatches to generate_multi_node_script."""
    cluster = _make_cluster()
    resources = _make_resources(nodes=2)
    _generate_script(cluster, resources, "job-1", "sft")
    mock_multi.assert_called_once_with(cluster, resources, "job-1", "sft")
