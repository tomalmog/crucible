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
        result = _suggest_modules(session, result)

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
    if not result.slurm_ok:
        return result

    gpu_types: list[str] = []

    # Method 1: parse GRES — gpu:type:count format
    stdout, _, code = session.execute(
        "sinfo --noheader -o '%G' | sort -u", timeout=15,
    )
    if code == 0:
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line or line == "(null)":
                continue
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "gpu":
                candidate = parts[1].split("(")[0]
                if candidate and candidate not in gpu_types:
                    gpu_types.append(candidate)

    # Method 2: if GRES didn't have named types, check node features
    if not gpu_types:
        stdout, _, code = session.execute(
            "sinfo --noheader -o '%f' | tr ',' '\\n' | sort -u", timeout=15,
        )
        if code == 0:
            gpu_keywords = ("gpu", "a100", "a40", "v100", "h100", "l40",
                            "t4", "rtx", "titan", "p100", "a10", "l4")
            for line in stdout.strip().splitlines():
                feat = line.strip().lower()
                if not feat or feat == "(null)":
                    continue
                if any(kw in feat for kw in gpu_keywords):
                    name = line.strip()
                    if name not in gpu_types:
                        gpu_types.append(name)

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
