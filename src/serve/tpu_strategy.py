"""TPU parallelism strategy using torch_xla.

Implements the ParallelismStrategy protocol for Google Cloud TPU
training workflows using XLA-based model wrapping and checkpointing.
"""

from __future__ import annotations

from typing import Any

from core.errors import ForgeDependencyError, ForgeDistributedError
from serve.tpu_setup import import_xla, init_xla_mesh, resolve_tpu_device


class TpuStrategy:
    """ParallelismStrategy implementation for TPU devices via torch_xla.

    Wraps models for XLA execution, builds optimizers compatible with
    XLA devices, and provides checkpoint save/load through XLA serialization.
    """

    def __init__(self) -> None:
        self._xla = import_xla()
        self._torch = _import_torch()

    def wrap_model(self, model: Any, device: Any) -> Any:
        """Move model to TPU XLA device and initialize mesh.

        Args:
            model: The PyTorch model to wrap.
            device: Ignored; TPU device is resolved internally.

        Returns:
            The model placed on the XLA device.
        """
        xla_device = resolve_tpu_device(self._xla)
        init_xla_mesh(self._xla)
        model = model.to(xla_device)
        return model

    def build_optimizer(
        self, model: Any, lr: float, weight_decay: float,
    ) -> Any:
        """Build an AdamW optimizer for the TPU-placed model.

        Args:
            model: The wrapped model on TPU.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            An optimizer instance.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        return self._torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay,
        )

    def save_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Save model and optimizer state via XLA serialization.

        Args:
            model: The wrapped model.
            optimizer: The optimizer.
            path: File path for the checkpoint.

        Raises:
            ForgeDistributedError: If checkpoint save fails.
        """
        try:
            xm = self._xla
            _mark_step_if_available(xm)
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            _xla_save(xm, self._torch, state, path)
        except Exception as error:
            raise ForgeDistributedError(
                f"Failed to save TPU checkpoint: {error}."
            ) from error

    def load_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Load model and optimizer state from an XLA checkpoint.

        Args:
            model: The wrapped model.
            optimizer: The optimizer.
            path: File path of the checkpoint.

        Raises:
            ForgeDistributedError: If checkpoint load fails.
        """
        try:
            xla_device = resolve_tpu_device(self._xla)
            state = self._torch.load(
                path, map_location=xla_device, weights_only=True,
            )
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as error:
            raise ForgeDistributedError(
                f"Failed to load TPU checkpoint: {error}."
            ) from error

    def strategy_name(self) -> str:
        """Return the canonical name of this strategy."""
        return "tpu"


def _mark_step_if_available(xla_module: Any) -> None:
    """Call xla mark_step if available to sync before save."""
    mark_step = getattr(xla_module, "mark_step", None)
    if callable(mark_step):
        mark_step()


def _xla_save(xla_module: Any, torch_module: Any, state: dict[str, Any], path: str) -> None:
    """Save state using XLA save if available, falling back to torch.save."""
    xla_save_fn = getattr(xla_module, "save", None)
    if callable(xla_save_fn):
        xla_save_fn(state, path)
    else:
        torch_module.save(state, path)


def _import_torch() -> Any:
    """Import torch dependency required by TPU strategy."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "TPU strategy requires torch. "
            "Install torch to use TPU training."
        ) from error
    return torch
