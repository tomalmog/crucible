"""SSH-based validation of remote Slurm cluster readiness.

Checks Python, PyTorch, CUDA, Slurm availability, discovers partitions
and GPU types, and suggests module loads.

Every check that relies on the crucible conda env must prefix its command
with the conda init + activate snippet because each ``session.execute``
runs in its own shell.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from core.slurm_types import ClusterConfig, ClusterValidationResult
from serve.remote_env_setup import CONDA_ACTIVATE, ensure_remote_env
from serve.ssh_connection import SshSession


def validate_ssh_cluster(cluster: ClusterConfig) -> ClusterValidationResult:
    """Validate an SSH cluster's readiness.

    For Docker SSH: checks Docker and GPU access.
    For bare SSH: tries conda provisioning first; if conda is unavailable
    (cloud GPU providers), checks system Python for torch directly.
    """
    result = ClusterValidationResult(cluster_name=cluster.name)
    errors: list[str] = []

    with SshSession(cluster) as session:
        if cluster.docker_image:
            result = _check_ssh_python(session, cluster, result, errors)
            result = _check_docker(session, result, errors)
            if result.docker_ok:
                result = _check_docker_gpu(session, cluster, result, errors)
        else:
            result = _validate_bare_ssh_env(
                session, cluster, result, errors,
            )

    return replace(result, errors=tuple(errors))


def _validate_bare_ssh_env(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Validate bare SSH environment — conda or system Python.

    If conda is not installed, installs Miniconda automatically.
    """
    try:
        ensure_remote_env(session)
    except Exception:
        # Conda not available — install it, then retry
        try:
            from serve.ssh_submit_helpers import _install_miniconda
            _install_miniconda(session)
            ensure_remote_env(session)
        except Exception as exc:
            errors.append(f"Failed to provision environment: {exc}")
            return result

    result = _check_python(session, cluster, result, errors)
    result = _check_torch(session, cluster, result, errors)
    return result


def validate_cluster(cluster: ClusterConfig) -> ClusterValidationResult:
    """Validate a remote cluster's readiness for Crucible jobs.

    Opens an SSH connection, ensures the crucible conda env exists,
    runs diagnostic commands, and returns a structured validation result.

    Args:
        cluster: Cluster configuration to validate.

    Returns:
        ClusterValidationResult with per-check status.
    """
    result = ClusterValidationResult(cluster_name=cluster.name)
    errors: list[str] = []

    with SshSession(cluster) as session:
        try:
            ensure_remote_env(session)
        except Exception as exc:
            errors.append(f"Failed to provision crucible env: {exc}")
            return replace(result, errors=tuple(errors))

        result = _check_python(session, cluster, result, errors)
        result = _check_torch(session, cluster, result, errors)
        result = _check_slurm(session, result, errors)
        result = _discover_gpu_types(session, result)

    return replace(result, errors=tuple(errors))


def _env_prefix(cluster: ClusterConfig) -> str:
    """Build a shell prefix that loads modules and activates the crucible env."""
    parts: list[str] = list(cluster.module_loads)
    parts.append(CONDA_ACTIVATE)
    return " && ".join(parts)


def _check_python(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Python is accessible inside the crucible env."""
    prefix = _env_prefix(cluster)
    py = cluster.python_path
    stdout, stderr, code = session.execute(
        f"{prefix} && {py} --version", timeout=30,
    )
    if code == 0:
        version = stdout.strip() or stderr.strip()
        return replace(result, python_ok=True, python_version=version)
    errors.append(f"Python not found at '{py}': {stderr.strip()}")
    return result


def _check_torch(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if PyTorch and CUDA are available inside the crucible env."""
    prefix = _env_prefix(cluster)
    py = cluster.python_path
    script = (
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda={torch.cuda.is_available()}'); "
        "v=torch.version.cuda or ''; "
        "print(f'cuda_ver={v}')"
    )
    stdout, stderr, code = session.execute(
        f'{prefix} && {py} -c "{script}"', timeout=30,
    )
    if code != 0:
        errors.append(f"PyTorch not available: {stderr.strip()}")
        return result

    lines = stdout.strip().splitlines()
    info: dict[str, str] = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v

    torch_version = info.get("torch", "")
    cuda_available = info.get("cuda", "False") == "True"
    cuda_version = info.get("cuda_ver", "")

    return replace(
        result,
        torch_ok=True,
        torch_version=torch_version,
        cuda_ok=cuda_available,
        cuda_version=cuda_version,
    )


def _check_slurm(
    session: SshSession,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Slurm commands are available and discover partitions."""
    stdout, _, code = session.execute("sinfo --noheader -o '%P'", timeout=15)
    if code != 0:
        errors.append("Slurm is not available (sinfo not found).")
        return result
    partitions = tuple(
        p.rstrip("*") for p in stdout.strip().splitlines() if p.strip()
    )
    return replace(result, slurm_ok=True, partitions=partitions)


def _discover_gpu_types(
    session: SshSession,
    result: ClusterValidationResult,
) -> ClusterValidationResult:
    """Discover available GPU types from Slurm GRES and node features."""
    from serve.slurm_discovery import discover_gpu_types, suggest_modules
    result = discover_gpu_types(session, result)
    return suggest_modules(session, result)


def _check_ssh_python(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Python is accessible on the remote host (no conda env)."""
    py = cluster.python_path
    stdout, stderr, code = session.execute(f"{py} --version", timeout=15)
    if code == 0:
        version = stdout.strip() or stderr.strip()
        return replace(result, python_ok=True, python_version=version)
    errors.append(f"Python not found at '{py}': {stderr.strip()}")
    return result


def _check_docker(
    session: SshSession,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Docker is installed and accessible."""
    stdout, stderr, code = session.execute("docker --version", timeout=15)
    if code != 0:
        errors.append(f"Docker not found: {stderr.strip()}")
        return result
    version = stdout.strip()
    return replace(result, docker_ok=True, python_version=version)


def _check_docker_gpu(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Docker has GPU access via nvidia-smi."""
    from core.constants import DEFAULT_DOCKER_IMAGE

    image = cluster.docker_image or DEFAULT_DOCKER_IMAGE
    stdout, stderr, code = session.execute(
        f"docker run --rm --gpus all {image} nvidia-smi --query-gpu=name --format=csv,noheader",
        timeout=60,
    )
    if code != 0:
        errors.append(f"Docker GPU access failed: {stderr.strip()}")
        return result
    gpu_names = [g.strip() for g in stdout.strip().splitlines() if g.strip()]
    return replace(
        result,
        docker_gpu_ok=True,
        gpu_types=tuple(dict.fromkeys(gpu_names)),
    )


def update_cluster_validated(cluster: ClusterConfig) -> ClusterConfig:
    """Return a copy of the cluster with validated_at set to now."""
    return replace(
        cluster,
        validated_at=datetime.now(timezone.utc).isoformat(),
    )
