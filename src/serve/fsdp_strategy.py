"""FSDP parallelism strategy for large-model distributed training.

This module implements the ParallelismStrategy protocol using PyTorch
Fully Sharded Data Parallel (FSDP). It supports configurable sharding
strategies, auto-wrapping, and CPU offloading for memory efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import ForgeDependencyError, ForgeDistributedError
from serve.distributed_setup import get_rank

_VALID_SHARDING_STRATEGIES = frozenset({
    "full_shard",
    "shard_grad_op",
    "no_shard",
})


@dataclass(frozen=True)
class FsdpConfig:
    """Configuration for FSDP parallelism strategy.

    Attributes:
        sharding_strategy: One of 'full_shard', 'shard_grad_op', 'no_shard'.
        auto_wrap_min_params: Minimum parameter count for auto-wrapping.
        cpu_offload: Whether to offload parameters to CPU.
    """

    sharding_strategy: str = "full_shard"
    auto_wrap_min_params: int = 100_000
    cpu_offload: bool = False


def _import_torch() -> Any:
    """Import the torch module.

    Returns:
        The torch module.

    Raises:
        ForgeDependencyError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "FSDP strategy requires torch. "
            "Install torch to use FSDP training."
        ) from error
    return torch


def _import_fsdp(torch_module: Any) -> Any:
    """Import the FSDP module from torch.

    Args:
        torch_module: The torch module.

    Returns:
        The FSDP class.

    Raises:
        ForgeDependencyError: If FSDP is not available.
    """
    try:
        fsdp_mod = torch_module.distributed.fsdp
        fsdp_class = getattr(
            fsdp_mod, "FullyShardedDataParallel", None,
        )
    except AttributeError:
        fsdp_class = None
    if fsdp_class is None:
        raise ForgeDependencyError(
            "FSDP is not available in this PyTorch build. "
            "Upgrade to PyTorch >= 1.12 for FSDP support."
        )
    return fsdp_class


def _resolve_sharding_strategy(
    torch_module: Any, name: str,
) -> Any:
    """Resolve a sharding strategy enum from its string name.

    Args:
        torch_module: The torch module.
        name: One of 'full_shard', 'shard_grad_op', 'no_shard'.

    Returns:
        The ShardingStrategy enum value.

    Raises:
        ForgeDistributedError: If the name is invalid.
    """
    if name not in _VALID_SHARDING_STRATEGIES:
        raise ForgeDistributedError(
            f"Invalid sharding strategy '{name}'. "
            f"Supported: {sorted(_VALID_SHARDING_STRATEGIES)}."
        )
    mapping = {
        "full_shard": "FULL_SHARD",
        "shard_grad_op": "SHARD_GRAD_OP",
        "no_shard": "NO_SHARD",
    }
    try:
        strategy_enum = torch_module.distributed.fsdp.ShardingStrategy
        return getattr(strategy_enum, mapping[name])
    except AttributeError:
        raise ForgeDistributedError(
            f"ShardingStrategy.{mapping[name]} is not available."
        )


def _build_cpu_offload(torch_module: Any, enabled: bool) -> Any:
    """Build a CPUOffload configuration object.

    Args:
        torch_module: The torch module.
        enabled: Whether CPU offloading is enabled.

    Returns:
        A CPUOffload instance, or None if disabled.
    """
    if not enabled:
        return None
    try:
        offload_cls = torch_module.distributed.fsdp.CPUOffload
        return offload_cls(offload_params=True)
    except AttributeError:
        return None


def _build_auto_wrap_policy(
    torch_module: Any, min_params: int,
) -> Any:
    """Build an auto-wrap policy based on minimum parameter count.

    Args:
        torch_module: The torch module.
        min_params: Minimum parameters for a sub-module to be wrapped.

    Returns:
        A wrap policy callable, or None if unavailable.
    """
    try:
        from functools import partial
        wrap_mod = torch_module.distributed.fsdp.wrap
        policy_fn = getattr(
            wrap_mod, "size_based_auto_wrap_policy", None,
        )
        if policy_fn is None:
            return None
        return partial(policy_fn, min_num_params=min_params)
    except AttributeError:
        return None


class FsdpStrategy:
    """Fully Sharded Data Parallel parallelism strategy.

    Wraps a model with FSDP, supporting configurable sharding,
    auto-wrapping, and CPU offloading for large models.
    """

    def __init__(self, config: FsdpConfig | None = None) -> None:
        self._config = config or FsdpConfig()

    def wrap_model(self, model: Any, device: Any) -> Any:
        """Wrap a model with FSDP.

        Args:
            model: PyTorch model to wrap.
            device: Target device for this rank.

        Returns:
            The FSDP-wrapped model.
        """
        torch_module = _import_torch()
        fsdp_class = _import_fsdp(torch_module)
        sharding = _resolve_sharding_strategy(
            torch_module, self._config.sharding_strategy,
        )
        cpu_offload = _build_cpu_offload(
            torch_module, self._config.cpu_offload,
        )
        auto_wrap = _build_auto_wrap_policy(
            torch_module, self._config.auto_wrap_min_params,
        )
        kwargs: dict[str, Any] = {
            "sharding_strategy": sharding,
        }
        if cpu_offload is not None:
            kwargs["cpu_offload"] = cpu_offload
        if auto_wrap is not None:
            kwargs["auto_wrap_policy"] = auto_wrap
        model = model.to(device)
        return fsdp_class(model, **kwargs)

    def build_optimizer(
        self, model: Any, lr: float, weight_decay: float,
    ) -> Any:
        """Build an AdamW optimizer for the FSDP model.

        Args:
            model: The FSDP-wrapped model.
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
        """Save FSDP checkpoint with full state dict on rank 0.

        Args:
            model: The FSDP-wrapped model.
            optimizer: The optimizer.
            path: File path for the checkpoint.
        """
        torch_module = _import_torch()
        rank = get_rank(torch_module)
        model_state = _gather_fsdp_state_dict(torch_module, model)
        if rank != 0:
            return
        torch_module.save(
            {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Load checkpoint into FSDP model and optimizer.

        Args:
            model: The FSDP-wrapped model.
            optimizer: The optimizer.
            path: File path of the checkpoint.
        """
        torch_module = _import_torch()
        checkpoint = torch_module.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def strategy_name(self) -> str:
        """Return the strategy name.

        Returns:
            The string 'fsdp'.
        """
        return "fsdp"


def _gather_fsdp_state_dict(
    torch_module: Any, model: Any,
) -> dict[str, Any]:
    """Gather the full state dict from an FSDP model.

    Uses FSDP's full_state_dict_type context manager when available,
    otherwise falls back to model.state_dict().

    Args:
        torch_module: The torch module.
        model: The FSDP-wrapped model.

    Returns:
        The gathered model state dict.
    """
    try:
        fsdp_mod = torch_module.distributed.fsdp
        fsdp_class = fsdp_mod.FullyShardedDataParallel
        state_dict_type = getattr(
            fsdp_mod, "StateDictType", None,
        )
        if state_dict_type is not None:
            full_type = state_dict_type.FULL_STATE_DICT
            with fsdp_class.state_dict_type(model, full_type):
                return dict(model.state_dict())
    except (AttributeError, TypeError):
        pass
    return dict(model.state_dict())
