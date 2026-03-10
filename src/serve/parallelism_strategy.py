"""Parallelism strategy protocol and factory for distributed training.

This module defines the interface that all parallelism strategies (DDP,
FSDP, DeepSpeed) must implement, and provides a factory function to
resolve a strategy by name.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.errors import CrucibleDistributedError


@runtime_checkable
class ParallelismStrategy(Protocol):
    """Protocol defining the distributed training strategy interface.

    Each strategy wraps model distribution, optimizer construction,
    and checkpoint I/O behind a uniform API so the training runner
    can remain strategy-agnostic.
    """

    def wrap_model(self, model: Any, device: Any) -> Any:
        """Wrap a model for distributed training.

        Args:
            model: The PyTorch model to wrap.
            device: Target device for this rank.

        Returns:
            The wrapped model ready for distributed training.
        """
        ...

    def build_optimizer(
        self, model: Any, lr: float, weight_decay: float,
    ) -> Any:
        """Build an optimizer for the wrapped model.

        Args:
            model: The wrapped model.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            An optimizer instance.
        """
        ...

    def save_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Save model and optimizer state to a checkpoint.

        Args:
            model: The wrapped model.
            optimizer: The optimizer.
            path: File path for the checkpoint.
        """
        ...

    def load_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Load model and optimizer state from a checkpoint.

        Args:
            model: The wrapped model.
            optimizer: The optimizer.
            path: File path of the checkpoint.
        """
        ...

    def strategy_name(self) -> str:
        """Return the canonical name of this strategy.

        Returns:
            Strategy identifier string.
        """
        ...


_STRATEGY_REGISTRY: dict[str, str] = {
    "ddp": "serve.ddp_strategy",
    "fsdp": "serve.fsdp_strategy",
    "deepspeed": "serve.deepspeed_strategy",
    "tpu": "serve.tpu_strategy",
}


def resolve_parallelism_strategy(
    strategy_name: str,
) -> ParallelismStrategy:
    """Resolve a parallelism strategy by name.

    Args:
        strategy_name: One of 'ddp', 'fsdp', or 'deepspeed'.

    Returns:
        An instance implementing the ParallelismStrategy protocol.

    Raises:
        CrucibleDistributedError: If the strategy name is unknown.
    """
    name = strategy_name.lower().strip()
    if name == "ddp":
        from serve.ddp_strategy import DdpStrategy
        return DdpStrategy()
    if name == "fsdp":
        from serve.fsdp_strategy import FsdpStrategy
        return FsdpStrategy()
    if name == "deepspeed":
        from serve.deepspeed_strategy import DeepSpeedStrategy
        return DeepSpeedStrategy()
    if name == "tpu":
        from serve.tpu_strategy import TpuStrategy
        return TpuStrategy()
    raise CrucibleDistributedError(
        f"Unknown parallelism strategy '{strategy_name}'. "
        f"Supported strategies: {sorted(_STRATEGY_REGISTRY.keys())}."
    )
