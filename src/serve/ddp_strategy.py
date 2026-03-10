"""DDP parallelism strategy for multi-GPU distributed training.

This module implements the ParallelismStrategy protocol using PyTorch
DistributedDataParallel. It wraps model distribution, optimizer
construction, and checkpoint I/O for standard DDP workflows.
"""

from __future__ import annotations

from typing import Any

from core.errors import CrucibleDependencyError, CrucibleDistributedError
from serve.distributed_setup import get_rank


def _import_torch() -> Any:
    """Import the torch module.

    Returns:
        The torch module.

    Raises:
        CrucibleDependencyError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "DDP strategy requires torch. "
            "Install torch to use distributed training."
        ) from error
    return torch


class DdpStrategy:
    """DistributedDataParallel parallelism strategy.

    Wraps a model with torch DDP, builds a standard AdamW optimizer,
    and handles rank-aware checkpoint save/load.
    """

    def wrap_model(self, model: Any, device: Any) -> Any:
        """Wrap a model with DistributedDataParallel.

        Args:
            model: PyTorch model to wrap.
            device: Target device for this rank.

        Returns:
            The DDP-wrapped model.

        Raises:
            CrucibleDistributedError: If DDP is unavailable.
        """
        torch_module = _import_torch()
        model = model.to(device)
        nn_parallel = getattr(torch_module.nn, "parallel", None)
        if nn_parallel is None:
            raise CrucibleDistributedError(
                "torch.nn.parallel is unavailable for DDP wrapping."
            )
        ddp_class = getattr(nn_parallel, "DistributedDataParallel", None)
        if ddp_class is None:
            raise CrucibleDistributedError(
                "DistributedDataParallel is not available."
            )
        device_id = _extract_device_index(device)
        return ddp_class(model, device_ids=[device_id])

    def build_optimizer(
        self, model: Any, lr: float, weight_decay: float,
    ) -> Any:
        """Build an AdamW optimizer for the DDP model.

        Args:
            model: The DDP-wrapped model.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            An AdamW optimizer instance.
        """
        torch_module = _import_torch()
        return torch_module.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

    def save_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Save checkpoint on rank 0 only.

        Args:
            model: The DDP-wrapped model.
            optimizer: The optimizer.
            path: File path for the checkpoint.
        """
        torch_module = _import_torch()
        rank = get_rank(torch_module)
        if rank != 0:
            return
        inner = getattr(model, "module", model)
        torch_module.save(
            {
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Load a checkpoint into model and optimizer.

        Args:
            model: The DDP-wrapped model.
            optimizer: The optimizer.
            path: File path of the checkpoint.
        """
        torch_module = _import_torch()
        checkpoint = torch_module.load(path, weights_only=True)
        inner = getattr(model, "module", model)
        inner.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def strategy_name(self) -> str:
        """Return the strategy name.

        Returns:
            The string 'ddp'.
        """
        return "ddp"


def _extract_device_index(device: Any) -> int:
    """Extract numeric device index from a torch device.

    Args:
        device: A torch device or string like 'cuda:0'.

    Returns:
        Integer device index, defaulting to 0.
    """
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    device_str = str(device)
    if ":" in device_str:
        try:
            return int(device_str.split(":")[-1])
        except ValueError:
            return 0
    return 0
