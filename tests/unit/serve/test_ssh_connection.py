"""Unit tests for SSH session wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(name="test", host="test.example.com", user="testuser")


def test_client_property_not_connected_raises() -> None:
    """Accessing client before connect raises ForgeRemoteError."""
    session = SshSession(_make_cluster())
    with pytest.raises(ForgeRemoteError, match="not connected"):
        _ = session.client


def test_context_manager_connects_and_closes() -> None:
    """__enter__ calls connect and __exit__ calls close."""
    session = SshSession(_make_cluster())
    with patch.object(session, "connect") as mock_connect, \
         patch.object(session, "close") as mock_close:
        with session:
            mock_connect.assert_called_once()
        mock_close.assert_called_once()


def test_execute_returns_stdout_stderr_exitcode() -> None:
    """execute() returns the (stdout, stderr, exit_code) tuple."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    stdout_ch = MagicMock()
    stderr_ch = MagicMock()
    stdout_ch.read.return_value = b"hello\n"
    stderr_ch.read.return_value = b"warn\n"
    stdout_ch.channel.recv_exit_status.return_value = 0
    mock_client.exec_command.return_value = (MagicMock(), stdout_ch, stderr_ch)
    session._client = mock_client

    result = session.execute("echo hello")
    assert result == ("hello\n", "warn\n", 0)


def test_execute_raises_on_error() -> None:
    """execute() wraps exceptions in ForgeRemoteError."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    mock_client.exec_command.side_effect = OSError("broken pipe")
    session._client = mock_client

    with pytest.raises(ForgeRemoteError, match="Remote command failed"):
        session.execute("ls")


def test_upload_calls_sftp_put() -> None:
    """upload() opens SFTP and calls put with correct paths."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    session._client = mock_client

    session.upload(Path("/tmp/local.bin"), "/remote/local.bin")
    mock_sftp.put.assert_called_once_with("/tmp/local.bin", "/remote/local.bin")


def test_upload_raises_on_sftp_error() -> None:
    """upload() wraps SFTP exceptions in ForgeRemoteError."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_sftp.put.side_effect = OSError("disk full")
    mock_client.open_sftp.return_value = mock_sftp
    session._client = mock_client

    with pytest.raises(ForgeRemoteError, match="SFTP upload failed"):
        session.upload(Path("/tmp/local.bin"), "/remote/local.bin")


def test_download_calls_sftp_get() -> None:
    """download() opens SFTP and calls get with correct paths."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    session._client = mock_client
    local = MagicMock(spec=Path)
    local.__str__ = lambda self: "/tmp/downloaded.bin"
    local.parent = MagicMock()

    session.download("/remote/file.bin", local)
    mock_sftp.get.assert_called_once_with("/remote/file.bin", "/tmp/downloaded.bin")


def test_download_creates_parent_dirs() -> None:
    """download() creates parent directories before fetching."""
    session = SshSession(_make_cluster())
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    session._client = mock_client
    local = MagicMock(spec=Path)
    local.__str__ = lambda self: "/tmp/sub/dir/file.bin"
    parent = MagicMock()
    local.parent = parent

    session.download("/remote/file.bin", local)
    parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_mkdir_p_executes_command() -> None:
    """mkdir_p() delegates to execute with 'mkdir -p' command."""
    session = SshSession(_make_cluster())
    with patch.object(session, "execute") as mock_execute:
        session.mkdir_p("/remote/new/dir")
        mock_execute.assert_called_once_with("mkdir -p /remote/new/dir")


def test_tail_last_returns_stdout() -> None:
    """tail_last() executes tail command and returns stdout."""
    session = SshSession(_make_cluster())
    with patch.object(
        session, "execute", return_value=("line1\nline2\n", "", 0),
    ) as mock_execute:
        result = session.tail_last("/var/log/train.log", lines=50)
        mock_execute.assert_called_once_with("tail -n 50 /var/log/train.log")
    assert result == "line1\nline2\n"
