"""Unit tests for remote conda env auto-provisioning."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.errors import ForgeRemoteError
from serve.remote_env_setup import ensure_remote_env


def _make_session(responses: list[tuple[str, str, int]]) -> MagicMock:
    """Build a mock SshSession with scripted execute() responses."""
    session = MagicMock()
    session.execute = MagicMock(side_effect=responses)
    return session


class TestEnvExists:
    """Tests for the fast-path when the env already exists."""

    def test_skips_creation_when_env_present(self) -> None:
        session = _make_session([
            ("forge                    /home/user/.conda/envs/forge\n", "", 0),
            ("torch=2.6.0\n", "", 0),  # torch check
        ])
        ensure_remote_env(session)
        assert session.execute.call_count == 2
        cmd = session.execute.call_args_list[0].args[0]
        assert "conda env list" in cmd

    def test_detects_env_among_multiple(self) -> None:
        output = (
            "base                  *  /opt/conda\n"
            "forge                    /opt/conda/envs/forge\n"
            "other                    /opt/conda/envs/other\n"
        )
        session = _make_session([
            (output, "", 0),
            ("torch=2.6.0\n", "", 0),  # torch check
        ])
        ensure_remote_env(session)
        assert session.execute.call_count == 2


class TestEnvCreation:
    """Tests for env creation when it doesn't exist."""

    def test_creates_env_when_missing(self) -> None:
        session = _make_session([
            ("base                  *  /opt/conda\n", "", 0),  # env list
            ("", "", 0),  # conda create
            ("", "", 1),  # nvidia-smi (no GPU on login node)
            ("", "", 1),  # srun nvidia-smi (fallback)
            ("", "", 0),  # torch install (cu124 default)
            ("", "", 0),  # pip install remaining
        ])
        ensure_remote_env(session)
        calls = session.execute.call_args_list
        assert "conda create -n forge python=3.11 -y" in calls[1].args[0]
        torch_call = next(c for c in calls if "torch" in c.args[0] and "pip" in c.args[0])
        assert "cu124" in torch_call.args[0]
        pip_call = next(c for c in calls if "pyyaml" in c.args[0])
        assert "pyyaml matplotlib tokenizers" in pip_call.args[0]

    def test_creates_env_when_list_empty(self) -> None:
        session = _make_session([
            ("", "", 0),  # env list — no envs
            ("", "", 0),  # conda create
            ("", "", 1),  # nvidia-smi
            ("", "", 1),  # srun nvidia-smi
            ("", "", 0),  # torch install
            ("", "", 0),  # pip install remaining
        ])
        ensure_remote_env(session)
        assert session.execute.call_count == 6

    def test_commands_include_conda_init(self) -> None:
        """Conda commands should source conda.sh init first."""
        session = _make_session([
            ("", "", 0),  # env list
            ("", "", 0),  # conda create
            ("", "", 1),  # nvidia-smi
            ("", "", 1),  # srun nvidia-smi
            ("", "", 0),  # torch install
            ("", "", 0),  # pip install remaining
        ])
        ensure_remote_env(session)
        for c in session.execute.call_args_list:
            cmd = c.args[0]
            # nvidia-smi and srun calls don't need conda init
            if "nvidia-smi" in cmd or "srun" in cmd:
                continue
            assert "profile.d/conda.sh" in cmd


class TestErrors:
    """Tests for error handling."""

    def test_raises_when_conda_unavailable(self) -> None:
        session = _make_session([
            ("", "conda: command not found", 127),
        ])
        with pytest.raises(ForgeRemoteError, match="conda is not available"):
            ensure_remote_env(session)

    def test_raises_when_create_fails(self) -> None:
        session = _make_session([
            ("base  /opt/conda\n", "", 0),
            ("", "PackagesNotFoundError", 1),
        ])
        with pytest.raises(ForgeRemoteError, match="conda create failed"):
            ensure_remote_env(session)

    def test_raises_when_torch_install_fails(self) -> None:
        session = _make_session([
            ("base  /opt/conda\n", "", 0),
            ("", "", 0),  # conda create OK
            ("", "", 1),  # nvidia-smi
            ("", "", 1),  # srun nvidia-smi
            ("", "ERROR: No matching distribution", 1),  # torch install
        ])
        with pytest.raises(ForgeRemoteError, match="torch install failed"):
            ensure_remote_env(session)

    def test_raises_when_pip_install_fails(self) -> None:
        session = _make_session([
            ("base  /opt/conda\n", "", 0),
            ("", "", 0),  # conda create OK
            ("", "", 1),  # nvidia-smi
            ("", "", 1),  # srun nvidia-smi
            ("", "", 0),  # torch install OK
            ("", "ERROR: No matching distribution", 1),  # pip install
        ])
        with pytest.raises(ForgeRemoteError, match="pip install failed"):
            ensure_remote_env(session)
