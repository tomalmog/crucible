"""Unit tests for remote HuggingFace model downloading via SSH."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig
from serve.remote_hub_download import (
    _build_remote_model_path,
    _ensure_hf_hub_installed,
    _run_snapshot_download,
    download_model_to_cluster,
)


def _make_session(responses: list[tuple[str, str, int]]) -> MagicMock:
    """Build a mock SshSession with scripted execute() responses."""
    session = MagicMock()
    session.execute = MagicMock(side_effect=responses)
    return session


# -- _build_remote_model_path ------------------------------------------------


def test_build_remote_model_path_sanitizes_repo_id() -> None:
    """Slashes and special chars in repo_id are replaced with underscores."""
    result = _build_remote_model_path("/scratch", "meta-llama/Llama-2-7b")
    assert result == "/scratch/models/meta-llama_Llama-2-7b"


# -- _ensure_hf_hub_installed ------------------------------------------------


def test_ensure_hf_hub_installed_skips_when_present() -> None:
    """No install attempt when the import check succeeds."""
    session = _make_session([("hf_ok", "", 0)])
    _ensure_hf_hub_installed(session)
    assert session.execute.call_count == 1


def test_ensure_hf_hub_installed_installs_when_missing() -> None:
    """Falls back to pip install when the import check fails."""
    session = _make_session([("", "ModuleNotFoundError", 1), ("", "", 0)])
    _ensure_hf_hub_installed(session)
    assert session.execute.call_count == 2


def test_ensure_hf_hub_installed_raises_on_install_failure() -> None:
    """Raises ForgeRemoteError when pip install also fails."""
    session = _make_session([("", "", 1), ("", "pip error", 1)])
    with pytest.raises(ForgeRemoteError, match="Failed to install huggingface_hub"):
        _ensure_hf_hub_installed(session)


# -- _run_snapshot_download ---------------------------------------------------


def test_run_snapshot_download_success() -> None:
    """No error when download reports the target path."""
    session = _make_session([("downloaded_to=/scratch/models/test", "", 0)])
    _run_snapshot_download(session, "org/model", "/scratch/models/test", None)


def test_run_snapshot_download_raises_on_failure() -> None:
    """Raises ForgeRemoteError when the remote command exits non-zero."""
    session = _make_session([("", "network error", 1)])
    with pytest.raises(ForgeRemoteError, match="Remote download of org/model failed"):
        _run_snapshot_download(session, "org/model", "/scratch/models/test", None)


def test_run_snapshot_download_raises_when_path_missing() -> None:
    """Raises ForgeRemoteError when stdout lacks the expected marker."""
    session = _make_session([("some other output", "", 0)])
    with pytest.raises(ForgeRemoteError, match="path not reported"):
        _run_snapshot_download(session, "org/model", "/scratch/models/test", None)


# -- download_model_to_cluster -----------------------------------------------


@patch("serve.remote_hub_download.ensure_remote_env")
@patch("serve.remote_hub_download.SshSession")
@patch("serve.remote_hub_download.load_cluster")
def test_download_model_to_cluster_returns_remote_path(
    mock_load_cluster: MagicMock,
    mock_ssh_cls: MagicMock,
    mock_ensure_env: MagicMock,
) -> None:
    """Returns the constructed remote model path after a successful download."""
    mock_load_cluster.return_value = ClusterConfig(
        name="test", host="hpc", user="jdoe", remote_workspace="/scratch",
    )
    session = _make_session([
        ("hf_ok", "", 0),                                # _ensure_hf_hub_installed
        ("downloaded_to=/scratch/models/test", "", 0),    # _run_snapshot_download
    ])
    session.mkdir_p = MagicMock()
    mock_ssh_cls.return_value.__enter__ = MagicMock(return_value=session)
    mock_ssh_cls.return_value.__exit__ = MagicMock(return_value=False)

    result = download_model_to_cluster(
        data_root="/data", repo_id="org/model", cluster_name="test",
    )
    assert result == "/scratch/models/org_model"
