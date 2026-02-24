"""Accelerator-aware torch device selection helpers.

This module centralizes runtime device selection for training and inference.
It keeps accelerator detection consistent across serve components.
"""

from __future__ import annotations

from typing import Any


def resolve_execution_device(torch_module: Any) -> Any:
    """Resolve the preferred torch device for execution."""
    if is_tpu_available():
        from serve.tpu_setup import import_xla, resolve_tpu_device
        return resolve_tpu_device(import_xla())
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device("cuda")
    if is_mps_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def resolve_device_for_rank(torch_module: Any, rank: int) -> Any:
    """Resolve device for a specific DDP rank.

    Assigns cuda:<rank> when CUDA is available, otherwise falls back to CPU.

    Args:
        torch_module: Imported torch module.
        rank: Process rank index.

    Returns:
        Torch device for the given rank.
    """
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device(f"cuda:{rank}")
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
