"""Accelerator-aware torch device selection helpers.

This module centralizes runtime device selection for training and inference.
It keeps accelerator detection consistent across serve components.
"""

from __future__ import annotations

import os
from typing import Any

from core.errors import ForgeServeError


def _cuda_expected() -> bool:
    """Return True when the environment indicates GPUs should be available.

    Detects Slurm GPU jobs, CUDA_VISIBLE_DEVICES set, or NVIDIA driver present.
    """
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return True
    if os.environ.get("NVIDIA_VISIBLE_DEVICES"):
        return True
    return False


def resolve_execution_device(torch_module: Any) -> Any:
    """Resolve the preferred torch device for execution.

    Raises:
        ForgeServeError: If GPUs are expected (Slurm job, CUDA_VISIBLE_DEVICES set)
            but CUDA is not available. Never silently falls back to CPU.
    """
    if is_tpu_available():
        from serve.tpu_setup import import_xla, resolve_tpu_device
        return resolve_tpu_device(import_xla())
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device("cuda")
    if _cuda_expected():
        raise ForgeServeError(
            "CUDA GPUs were expected but torch.cuda.is_available() returned False. "
            "Possible causes: (1) this node has a broken GPU driver (cuInit error 999), "
            "try resubmitting to get a different node; "
            "(2) PyTorch CUDA version doesn't match the driver. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}, "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'unset')}."
        )
    if is_mps_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def resolve_device_for_rank(torch_module: Any, rank: int) -> Any:
    """Resolve device for a specific DDP rank.

    Assigns cuda:<rank> when CUDA is available, otherwise fails if GPUs expected.

    Args:
        torch_module: Imported torch module.
        rank: Process rank index.

    Returns:
        Torch device for the given rank.

    Raises:
        ForgeServeError: If GPUs are expected but CUDA is not available.
    """
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device(f"cuda:{rank}")
    if _cuda_expected():
        raise ForgeServeError(
            "CUDA GPUs were expected but torch.cuda.is_available() returned False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}. "
            "Reinstall torch with the correct CUDA version."
        )
    return torch_module.device("cpu")


def is_tpu_available() -> bool:
    """Return True when torch_xla is installed and TPU devices are detected."""
    from serve.tpu_setup import detect_tpu_availability
    return detect_tpu_availability()


def is_mps_available(torch_module: Any) -> bool:
    """Return True when torch reports MPS backend support and availability."""
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return False
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    probe = getattr(mps_backend, "is_available", None)
    if not callable(probe):
        return False
    return bool(probe())
