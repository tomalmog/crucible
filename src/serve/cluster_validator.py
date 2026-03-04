"""SSH-based validation of remote Slurm cluster readiness.

Checks Python, PyTorch, CUDA, Slurm availability, discovers partitions
and GPU types, and suggests module loads.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from core.slurm_types import ClusterConfig, ClusterValidationResult
from serve.ssh_connection import SshSession


def validate_cluster(cluster: ClusterConfig) -> ClusterValidationResult:
    """Validate a remote cluster's readiness for Forge jobs.

    Opens an SSH connection, runs diagnostic commands, and returns
    a structured validation result.

    Args:
        cluster: Cluster configuration to validate.

    Returns:
        ClusterValidationResult with per-check status.
    """
    result = ClusterValidationResult(cluster_name=cluster.name)
    errors: list[str] = []

    with SshSession(cluster) as session:
        # Run module loads first if configured
        if cluster.module_loads:
            module_cmd = " && ".join(cluster.module_loads)
            session.execute(module_cmd, timeout=30)

        result = _check_python(session, cluster, result, errors)
        result = _check_torch(session, cluster, result, errors)
        result = _check_slurm(session, result, errors)
        result = _discover_gpu_types(session, result)
        result = _suggest_modules(session, result)

    return replace(result, errors=tuple(errors))


def _check_python(
    session: SshSession,
    cluster: ClusterConfig,
    result: ClusterValidationResult,
    errors: list[str],
) -> ClusterValidationResult:
    """Check if Python is accessible on the remote."""
    py = cluster.python_path
    stdout, stderr, code = session.execute(
        f"{py} --version", timeout=15,
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
    """Check if PyTorch and CUDA are available."""
    py = cluster.python_path
    script = (
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda={torch.cuda.is_available()}'); "
        "print(f'cuda_ver={torch.version.cuda or \"\"}')"
    )
    stdout, stderr, code = session.execute(
        f'{py} -c "{script}"', timeout=30,
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
    """Discover available GPU types from Slurm GRES configuration."""
    if not result.slurm_ok:
        return result
    stdout, _, code = session.execute(
        "sinfo --noheader -o '%G' | sort -u", timeout=15,
    )
    if code != 0:
        return result

    gpu_types: list[str] = []
    for line in stdout.strip().splitlines():
        # GRES format: gpu:type:count or gpu:count
        parts = line.strip().split(":")
        if len(parts) >= 2 and parts[0] == "gpu" and not parts[1].isdigit():
            if parts[1] not in gpu_types:
                gpu_types.append(parts[1])
    return replace(result, gpu_types=tuple(gpu_types))


def _suggest_modules(
    session: SshSession,
    result: ClusterValidationResult,
) -> ClusterValidationResult:
    """Suggest module load commands if modules system is available."""
    stdout, _, code = session.execute("module avail 2>&1 | head -20", timeout=15)
    if code != 0:
        return result

    suggestions: list[str] = []
    if not result.torch_ok:
        # Look for cuda and python modules
        for line in stdout.splitlines():
            lower = line.lower()
            if "cuda" in lower:
                suggestions.append(f"module load {line.strip().split()[0]}")
            if "python" in lower:
                suggestions.append(f"module load {line.strip().split()[0]}")
    return replace(result, module_suggestions=tuple(suggestions))


def update_cluster_validated(cluster: ClusterConfig) -> ClusterConfig:
    """Return a copy of the cluster with validated_at set to now."""
    return replace(
        cluster,
        validated_at=datetime.now(timezone.utc).isoformat(),
    )
