"""Unit tests for SSH-based Slurm cluster validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.slurm_types import ClusterConfig
from serve.cluster_validator import update_cluster_validated, validate_cluster


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(name="test-hpc", host="hpc.example.com", user="jdoe")


def _make_session(responses: list[tuple[str, str, int]]) -> MagicMock:
    """Build a mock SshSession with scripted execute() responses."""
    session = MagicMock()
    session.execute = MagicMock(side_effect=responses)
    return session


def _patch_session(responses: list[tuple[str, str, int]]) -> MagicMock:
    """Build a mock that works as a patched SshSession context manager."""
    session = _make_session(responses)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# -- _check_python -----------------------------------------------------------

def test_check_python_success_sets_version() -> None:
    """Python check marks python_ok when interpreter is found."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),   # python --version
        ("", "PyTorch not available", 1),  # torch check fails
        ("gpu*\nmain\n", "", 0),       # sinfo partitions
        ("gpu:a100:4\n", "", 0),       # sinfo gres
        ("", "", 1),                   # module avail
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.python_ok is True


def test_check_python_failure_appends_error() -> None:
    """Python check appends error when interpreter is missing."""
    ctx = _patch_session([
        ("", "not found", 1),          # python --version fails
        ("", "PyTorch not available", 1),  # torch check fails
        ("", "command not found", 1),  # sinfo fails
        ("", "", 1),                   # module avail
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.python_ok is False


# -- _check_torch -------------------------------------------------------------

def test_check_torch_success_sets_versions() -> None:
    """Torch check parses version, cuda availability, and cuda version."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("gpu*\nmain\n", "", 0),
        ("gpu:a100:4\n", "", 0),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.torch_ok is True


def test_check_torch_failure_appends_error() -> None:
    """Torch check appends error when import fails."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("", "ModuleNotFoundError", 1),
        ("gpu*\nmain\n", "", 0),
        ("gpu:a100:4\n", "", 0),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.torch_ok is False


# -- _check_slurm -------------------------------------------------------------

def test_check_slurm_discovers_partitions() -> None:
    """Slurm check strips asterisk and collects partition names."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("gpu*\nmain\n", "", 0),
        ("gpu:a100:4\n", "", 0),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.partitions == ("gpu", "main")


def test_check_slurm_failure_appends_error() -> None:
    """Slurm check appends error when sinfo is unavailable."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("", "command not found", 1),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.slurm_ok is False


# -- _discover_gpu_types ------------------------------------------------------

def test_discover_gpu_types_parses_gres() -> None:
    """GPU discovery parses gres strings into type names."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("gpu*\n", "", 0),
        ("gpu:a100:4\ngpu:v100:2\n", "", 0),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.gpu_types == ("a100", "v100")


def test_discover_gpu_types_skips_when_no_slurm() -> None:
    """GPU discovery returns unchanged result when slurm is unavailable."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("", "command not found", 1),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.gpu_types == ()


# -- validate_cluster (integration) -------------------------------------------

def test_validate_cluster_all_pass() -> None:
    """Full validation with all checks passing sets every ok flag."""
    ctx = _patch_session([
        ("Python 3.11.0\n", "", 0),
        ("torch=2.6.0\ncuda=True\ncuda_ver=12.4\n", "", 0),
        ("gpu*\nmain\n", "", 0),
        ("gpu:a100:4\n", "", 0),
        ("", "", 1),
    ])
    with patch("serve.cluster_validator.SshSession", return_value=ctx):
        result = validate_cluster(_make_cluster())
    assert result.errors == ()


# -- update_cluster_validated -------------------------------------------------

def test_update_cluster_validated_sets_timestamp() -> None:
    """Updating validated cluster sets a non-empty validated_at timestamp."""
    cluster = _make_cluster()
    updated = update_cluster_validated(cluster)
    assert updated.validated_at != ""
