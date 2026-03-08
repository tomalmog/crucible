"""SSH session wrapper using paramiko for remote Slurm operations.

Provides persistent SSH connections with command execution, SFTP
file transfer, and streaming log reads.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig

if TYPE_CHECKING:
    import paramiko


def _require_paramiko() -> tuple[type, type, type]:
    """Import paramiko at runtime and return key classes."""
    try:
        import paramiko as pm
        return pm.SSHClient, pm.AutoAddPolicy, pm.RSAKey  # type: ignore[return-value]
    except ImportError as error:
        raise ForgeRemoteError(
            "paramiko is required for remote operations. "
            "Install with: pip install 'forge[remote]'"
        ) from error


class SshSession:
    """Persistent SSH session wrapping paramiko for Forge remote operations."""

    def __init__(self, cluster: ClusterConfig) -> None:
        self._cluster = cluster
        self._client: paramiko.SSHClient | None = None

    def connect(self) -> None:
        """Establish the SSH connection to the cluster."""
        SSHClient, AutoAddPolicy, _ = _require_paramiko()
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())

        # Load ~/.ssh/config for host aliases and settings
        ssh_config_path = os.path.expanduser("~/.ssh/config")
        hostname = self._cluster.host
        username = self._cluster.user
        config_key_filename: str | None = None
        if os.path.isfile(ssh_config_path):
            import paramiko as pm
            ssh_config = pm.SSHConfig.from_path(ssh_config_path)
            host_cfg = ssh_config.lookup(self._cluster.host)
            hostname = host_cfg.get("hostname", hostname)
            if not username:
                username = host_cfg.get("user", "")
            if not self._cluster.ssh_key_path and "identityfile" in host_cfg:
                files = host_cfg["identityfile"]
                config_key_filename = os.path.expanduser(files[0]) if files else None

        connect_kwargs: dict[str, object] = {
            "hostname": hostname,
            "username": username,
        }
        if self._cluster.ssh_key_path:
            key_path = os.path.expanduser(self._cluster.ssh_key_path)
            connect_kwargs["key_filename"] = key_path
        elif config_key_filename:
            connect_kwargs["key_filename"] = config_key_filename
        if self._cluster.password:
            connect_kwargs["password"] = self._cluster.password
        # Allow agent and gssapi fallback when no explicit auth is given
        connect_kwargs["allow_agent"] = True
        connect_kwargs["look_for_keys"] = True
        try:
            client.connect(**connect_kwargs)  # type: ignore[arg-type]
        except Exception as error:
            raise ForgeRemoteError(
                f"SSH connection to {hostname} failed: {error}"
            ) from error
        self._client = client

    def close(self) -> None:
        """Close the SSH connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> SshSession:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def client(self) -> paramiko.SSHClient:
        """Return the active SSH client, raising if not connected."""
        if self._client is None:
            raise ForgeRemoteError("SSH session is not connected.")
        return self._client

    def execute(self, command: str, timeout: int = 60) -> tuple[str, str, int]:
        """Execute a command on the remote host.

        Args:
            command: Shell command to execute.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        try:
            _, stdout_ch, stderr_ch = self.client.exec_command(
                command, timeout=timeout,
            )
            # Read data before exit status to avoid deadlock when
            # the remote buffers fill up.
            stdout = stdout_ch.read().decode("utf-8", errors="replace")
            stderr = stderr_ch.read().decode("utf-8", errors="replace")
            exit_code = stdout_ch.channel.recv_exit_status()
        except Exception as error:
            raise ForgeRemoteError(
                f"Remote command failed: {error}"
            ) from error
        return stdout, stderr, exit_code

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the remote host via SFTP.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path on the remote.
        """
        try:
            sftp = self.client.open_sftp()
            sftp.put(str(local_path), remote_path)
            sftp.close()
        except Exception as error:
            raise ForgeRemoteError(
                f"SFTP upload failed: {error}"
            ) from error

    def download(self, remote_path: str, local_path: Path) -> None:
        """Download a remote file to the local host via SFTP.

        Args:
            remote_path: Path on the remote host.
            local_path: Destination path on local machine.
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp = self.client.open_sftp()
            sftp.get(remote_path, str(local_path))
            sftp.close()
        except Exception as error:
            raise ForgeRemoteError(
                f"SFTP download failed: {error}"
            ) from error

    def upload_text(self, content: str, remote_path: str) -> None:
        """Write text content to a remote file via a temp file and SFTP."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", delete=False,
        ) as f:
            f.write(content)
            tmp_path = Path(f.name)
        try:
            self.upload(tmp_path, remote_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def mkdir_p(self, remote_path: str) -> None:
        """Create a directory (and parents) on the remote host."""
        self.execute(f"mkdir -p {remote_path}")

    def stream_command(
        self,
        command: str,
        timeout: int = 120,
    ) -> Generator[str, None, None]:
        """Execute a command and yield stdout chunks as they arrive.

        Args:
            command: Shell command to execute.
            timeout: Channel timeout in seconds.

        Yields:
            Raw stdout chunks (not split by line).

        Raises:
            ForgeRemoteError: On transport failure or non-zero exit.
        """
        try:
            transport = self.client.get_transport()
            if transport is None:
                raise ForgeRemoteError("SSH transport is not available.")
            channel = transport.open_session()
            channel.settimeout(float(timeout))
            channel.exec_command(command)
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    yield chunk
                else:
                    time.sleep(0.05)
            # Drain remaining stdout using blocking recv — the SSH
            # protocol can deliver exit status before all data arrives,
            # so recv_ready() alone would miss trailing chunks.
            while True:
                chunk_bytes = channel.recv(4096)
                if not chunk_bytes:
                    break
                yield chunk_bytes.decode("utf-8", errors="replace")
            exit_code = channel.recv_exit_status()
            stderr = ""
            while channel.recv_stderr_ready():
                stderr += channel.recv_stderr(4096).decode("utf-8", errors="replace")
            channel.close()
            if exit_code != 0:
                raise ForgeRemoteError(
                    f"Remote command exited with code {exit_code}: {stderr.strip()}"
                )
        except ForgeRemoteError:
            raise
        except Exception as error:
            raise ForgeRemoteError(
                f"Command streaming failed: {error}"
            ) from error

    def tail_follow(
        self,
        remote_path: str,
        initial_lines: int = 50,
    ) -> Generator[str, None, None]:
        """Stream lines from a remote file using tail -f.

        Args:
            remote_path: Path to the log file on the remote.
            initial_lines: Number of existing lines to show first.

        Yields:
            Individual lines from the remote log file.
        """
        command = f"tail -n {initial_lines} -f {remote_path}"
        try:
            transport = self.client.get_transport()
            if transport is None:
                raise ForgeRemoteError("SSH transport is not available.")
            channel = transport.open_session()
            channel.exec_command(command)
            buf = ""
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield line
            # Drain remaining
            while channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                buf += chunk
            if buf:
                yield buf
            channel.close()
        except ForgeRemoteError:
            raise
        except Exception as error:
            raise ForgeRemoteError(
                f"Log streaming failed: {error}"
            ) from error

    def tail_last(self, remote_path: str, lines: int = 100) -> str:
        """Fetch the last N lines from a remote file.

        Args:
            remote_path: Path to the file on the remote.
            lines: Number of trailing lines to retrieve.

        Returns:
            The last N lines as a single string.
        """
        stdout, _, _ = self.execute(f"tail -n {lines} {remote_path}")
        return stdout
