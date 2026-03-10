"""Distributed training process group initialization and teardown.

This module manages PyTorch distributed process groups for DDP training.
It wraps torch.distributed calls behind a clean interface that gracefully
handles non-distributed (single-GPU) execution contexts.
"""

from __future__ import annotations

from typing import Any

from core.errors import CrucibleDistributedError


def init_distributed(torch_module: Any, backend: str = "nccl") -> None:
    """Initialize the default process group for DDP training.

    Uses environment variables RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    (set by torchrun). Silently returns if already initialized.

    Args:
        torch_module: Imported torch module.
        backend: Communication backend (nccl, gloo, etc.).

    Raises:
        CrucibleDistributedError: If torch.distributed is unavailable.
    """
    dist = _get_dist_module(torch_module)
    if dist.is_initialized():
        return
    try:
        dist.init_process_group(backend=backend)
    except Exception as error:
        raise CrucibleDistributedError(
            f"Failed to initialize distributed process group with backend '{backend}': {error}. "
            "Ensure torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT."
        ) from error


def cleanup_distributed(torch_module: Any) -> None:
    """Destroy the process group if initialized.

    Args:
        torch_module: Imported torch module.
    """
    dist = _get_dist_module(torch_module)
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank(torch_module: Any) -> int:
    """Return current process rank, 0 if not distributed.

    Args:
        torch_module: Imported torch module.

    Returns:
        Integer rank of the current process.
    """
    dist = _get_dist_module(torch_module)
    if dist.is_initialized():
        return int(dist.get_rank())
    return 0


def get_world_size(torch_module: Any) -> int:
    """Return world size, 1 if not distributed.

    Args:
        torch_module: Imported torch module.

    Returns:
        Total number of distributed processes.
    """
    dist = _get_dist_module(torch_module)
    if dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def is_main_process(torch_module: Any) -> bool:
    """Return True if this is rank 0 (main process).

    Args:
        torch_module: Imported torch module.

    Returns:
        Whether this process is rank 0.
    """
    return get_rank(torch_module) == 0


def select_ddp_backend(torch_module: Any) -> str:
    """Select the best available DDP backend for current hardware.

    Prefers NCCL for CUDA, falls back to gloo otherwise.

    Args:
        torch_module: Imported torch module.

    Returns:
        Backend name string.
    """
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return "nccl"
    return "gloo"


def _get_dist_module(torch_module: Any) -> Any:
    """Get torch.distributed module.

    Args:
        torch_module: Imported torch module.

    Returns:
        The torch.distributed module.

    Raises:
        CrucibleDistributedError: If torch.distributed is unavailable.
    """
    dist = getattr(torch_module, "distributed", None)
    if dist is None:
        raise CrucibleDistributedError(
            "torch.distributed is not available. "
            "Ensure PyTorch is built with distributed support."
        )
    return dist
