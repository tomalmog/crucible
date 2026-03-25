"""Tests for SshRunner — mocks SSH to verify submit/cancel/get_state/logs/result."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleDockerError, CrucibleRemoteError
from core.job_types import JobSpec
from core.slurm_types import ClusterConfig
from serve.ssh_runner import SshRunner


def _make_docker_cluster() -> ClusterConfig:
    return ClusterConfig(
        name="test-cluster",
        host="gpu-node",
        user="testuser",
        docker_image="myimage:v1",
    )


def _make_bare_cluster() -> ClusterConfig:
    return ClusterConfig(
        name="bare-cluster",
        host="gpu-node",
        user="testuser",
        docker_image="",
    )


def _make_spec(cluster_name: str = "test-cluster") -> JobSpec:
    return JobSpec(
        job_type="sft",
        method_args={"dataset_name": "my-data"},
        backend="ssh",
        label="SFT test",
        cluster_name=cluster_name,
    )


def _write_cluster(tmp_path: Path, cluster: ClusterConfig) -> None:
    """Write a cluster config JSON to the temp data root."""
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir(exist_ok=True)
    cluster_dict = {
        "name": cluster.name, "host": cluster.host,
        "user": cluster.user, "docker_image": cluster.docker_image,
        "remote_workspace": "~/crucible-jobs",
    }
    (clusters_dir / f"{cluster.name}.json").write_text(json.dumps(cluster_dict))


@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Create a minimal data root with cluster configs."""
    _write_cluster(tmp_path, _make_docker_cluster())
    _write_cluster(tmp_path, _make_bare_cluster())
    return tmp_path


class TestSshRunnerKind:
    """Tests for SshRunner.kind property."""

    def test_kind_is_ssh(self) -> None:
        """Runner should report its kind as 'ssh'."""
        runner = SshRunner()
        assert runner.kind == "ssh"


class TestSshRunnerDockerSubmit:
    """Tests for SshRunner.submit() with Docker path."""

    @patch("serve.ssh_connection.SshSession", autospec=False)
    def test_submit_docker_creates_job_record(
        self, mock_session_cls: MagicMock, tmp_data_root: Path,
    ) -> None:
        """Successful Docker submit should create a job record on disk."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p.replace("~", "/home/test")
        mock_session.execute.return_value = ("abc123def456\n", "", 0)

        runner = SshRunner()
        record = runner.submit(tmp_data_root, _make_spec("test-cluster"))

        assert record.backend == "ssh"
        assert record.backend_job_id == "abc123def456"
        assert record.state == "running"
        assert record.job_type == "sft"

    @patch("serve.ssh_connection.SshSession", autospec=False)
    def test_submit_docker_failure_raises(
        self, mock_session_cls: MagicMock, tmp_data_root: Path,
    ) -> None:
        """Failed docker run should raise CrucibleDockerError."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p
        mock_session.execute.return_value = ("", "Error: no GPU", 1)

        runner = SshRunner()
        with pytest.raises(CrucibleDockerError):
            runner.submit(tmp_data_root, _make_spec("test-cluster"))


class TestSshRunnerBareSubmit:
    """Tests for SshRunner.submit() with bare SSH path."""

    @patch("serve.agent_bundler.build_agent_tarball")
    @patch("serve.ssh_submit_helpers.SshSession", autospec=False)
    def test_submit_bare_creates_job_record(
        self,
        mock_session_cls: MagicMock,
        mock_build_tarball: MagicMock,
        tmp_data_root: Path,
    ) -> None:
        """Successful bare SSH submit should create a job record with PID."""
        mock_tarball = tmp_data_root / "cache" / "agent" / "crucible-agent.tar.gz"
        mock_tarball.parent.mkdir(parents=True, exist_ok=True)
        mock_tarball.write_text("fake tarball")
        mock_build_tarball.return_value = mock_tarball

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p.replace("~", "/home/test")
        mock_session.execute.return_value = ("12345\n", "", 0)

        runner = SshRunner()
        record = runner.submit(tmp_data_root, _make_spec("bare-cluster"))

        assert record.backend == "ssh"
        assert record.backend_job_id == "12345"
        assert record.state == "running"
        assert record.job_type == "sft"

    @patch("serve.agent_bundler.build_agent_tarball")
    @patch("serve.ssh_submit_helpers.SshSession", autospec=False)
    def test_submit_bare_failure_raises(
        self,
        mock_session_cls: MagicMock,
        mock_build_tarball: MagicMock,
        tmp_data_root: Path,
    ) -> None:
        """Failed bare SSH launch should raise CrucibleRemoteError."""
        mock_tarball = tmp_data_root / "cache" / "agent" / "crucible-agent.tar.gz"
        mock_tarball.parent.mkdir(parents=True, exist_ok=True)
        mock_tarball.write_text("fake tarball")
        mock_build_tarball.return_value = mock_tarball

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p
        mock_session.execute.return_value = ("", "bash: python: not found", 1)

        runner = SshRunner()
        with pytest.raises(CrucibleRemoteError):
            runner.submit(tmp_data_root, _make_spec("bare-cluster"))


class TestSshRunnerBareSweepSubmit:
    """Tests for SshRunner.submit() with sweep on bare SSH."""

    @patch("serve.agent_bundler.build_agent_tarball")
    @patch("serve.ssh_submit_helpers.SshSession", autospec=False)
    def test_submit_bare_sweep_creates_sweep_record(
        self,
        mock_session_cls: MagicMock,
        mock_build_tarball: MagicMock,
        tmp_data_root: Path,
    ) -> None:
        """Bare SSH sweep submit should create a job record with is_sweep=True."""
        mock_tarball = tmp_data_root / "cache" / "agent" / "crucible-agent.tar.gz"
        mock_tarball.parent.mkdir(parents=True, exist_ok=True)
        mock_tarball.write_text("fake tarball")
        mock_build_tarball.return_value = mock_tarball

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p.replace("~", "/home/test")
        mock_session.execute.return_value = ("99999\n", "", 0)

        sweep_spec = JobSpec(
            job_type="sft",
            method_args={"model_name": "gpt2"},
            backend="ssh",
            label="Sweep test",
            cluster_name="bare-cluster",
            is_sweep=True,
            sweep_trials=(
                {"learning_rate": 1e-4},
                {"learning_rate": 5e-5},
            ),
        )
        runner = SshRunner()
        record = runner.submit(tmp_data_root, sweep_spec)

        assert record.backend == "ssh"
        assert record.is_sweep is True
        assert record.sweep_trial_count == 2
        assert record.state == "running"


class TestSshRunnerCancel:
    """Tests for SshRunner.cancel()."""

    @patch("serve.ssh_connection.SshSession", autospec=False)
    def test_cancel_docker_updates_state(
        self, mock_session_cls: MagicMock, tmp_data_root: Path,
    ) -> None:
        """Cancelling a Docker job should update the job state to 'cancelled'."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.resolve_path.side_effect = lambda p: p.replace("~", "/home/test")
        mock_session.execute.return_value = ("abc123def456\n", "", 0)

        runner = SshRunner()
        record = runner.submit(tmp_data_root, _make_spec("test-cluster"))

        mock_session.execute.return_value = ("abc123def456\n", "", 0)
        updated = runner.cancel(tmp_data_root, record.job_id)
        assert updated.state == "cancelled"
