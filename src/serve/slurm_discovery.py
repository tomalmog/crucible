"""Slurm GPU discovery and module suggestions.

Extracted from cluster_validator to keep file sizes manageable.
"""

from __future__ import annotations

from dataclasses import replace

from core.slurm_types import ClusterValidationResult
from serve.ssh_connection import SshSession


def discover_gpu_types(
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


def suggest_modules(
    session: SshSession,
    result: ClusterValidationResult,
) -> ClusterValidationResult:
    """Suggest module load commands if modules system is available."""
    stdout, _, code = session.execute("module avail 2>&1 | head -20", timeout=15)
    if code != 0:
        return result

    suggestions: list[str] = []
    if not result.torch_ok:
        for line in stdout.splitlines():
            lower = line.lower()
            if "cuda" in lower:
                suggestions.append(f"module load {line.strip().split()[0]}")
            if "python" in lower:
                suggestions.append(f"module load {line.strip().split()[0]}")
    return replace(result, module_suggestions=tuple(suggestions))
