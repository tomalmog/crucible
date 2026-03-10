"""DeepSpeed parallelism strategy for efficient large-model training.

This module implements the ParallelismStrategy protocol using Microsoft
DeepSpeed. It supports ZeRO optimization stages, CPU offloading for
optimizer and parameter states, and gradient clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleDistributedError

_VALID_ZERO_STAGES = frozenset({0, 1, 2, 3})


@dataclass(frozen=True)
class DeepSpeedConfig:
    """Configuration for DeepSpeed parallelism strategy.

    Attributes:
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3).
        offload_optimizer: Whether to offload optimizer state to CPU.
        offload_params: Whether to offload parameters to CPU.
        gradient_clipping: Maximum gradient norm for clipping.
    """

    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_params: bool = False
    gradient_clipping: float = 1.0


def build_deepspeed_config_dict(
    config: DeepSpeedConfig, lr: float, batch_size: int,
) -> dict[str, Any]:
    """Build a DeepSpeed JSON configuration dictionary.

    Args:
        config: The DeepSpeed configuration dataclass.
        lr: Learning rate for the optimizer.
        batch_size: Training micro-batch size.

    Returns:
        A dictionary suitable for deepspeed.initialize().

    Raises:
        CrucibleDistributedError: If the ZeRO stage is invalid.
    """
    if config.zero_stage not in _VALID_ZERO_STAGES:
        raise CrucibleDistributedError(
            f"Invalid ZeRO stage {config.zero_stage}. "
            f"Supported stages: {sorted(_VALID_ZERO_STAGES)}."
        )
    zero_config: dict[str, Any] = {
        "stage": config.zero_stage,
    }
    if config.offload_optimizer:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    if config.offload_params:
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_clipping": config.gradient_clipping,
        "zero_optimization": zero_config,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "fp16": {
            "enabled": False,
        },
    }
    return ds_config


def _import_deepspeed() -> Any:
    """Import the deepspeed module.

    Returns:
        The deepspeed module.

    Raises:
        CrucibleDependencyError: If deepspeed is not installed.
    """
    try:
        import deepspeed
    except ImportError as error:
        raise CrucibleDependencyError(
            "DeepSpeed strategy requires the deepspeed package. "
            "Install it with: pip install deepspeed"
        ) from error
    return deepspeed


class DeepSpeedStrategy:
    """DeepSpeed parallelism strategy.

    Wraps a model with DeepSpeed initialization, supporting ZeRO
    optimizer stages, CPU offloading, and gradient clipping.
    """

    def __init__(
        self, config: DeepSpeedConfig | None = None,
    ) -> None:
        self._config = config or DeepSpeedConfig()
        self._engine: Any = None

    def wrap_model(self, model: Any, device: Any) -> Any:
        """Wrap a model with DeepSpeed initialization.

        Args:
            model: PyTorch model to wrap.
            device: Target device for this rank.

        Returns:
            The DeepSpeed model engine.
        """
        deepspeed_mod = _import_deepspeed()
        ds_config = build_deepspeed_config_dict(
            self._config, lr=1e-4, batch_size=1,
        )
        engine, _, _, _ = deepspeed_mod.initialize(
            model=model,
            config=ds_config,
        )
        self._engine = engine
        return engine

    def build_optimizer(
        self, model: Any, lr: float, weight_decay: float,
    ) -> Any:
        """Return the DeepSpeed-managed optimizer.

        DeepSpeed manages the optimizer internally via its engine.
        This method updates the config and returns the engine's
        optimizer if available, otherwise returns a placeholder.

        Args:
            model: The DeepSpeed engine.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            The optimizer managed by DeepSpeed.
        """
        engine_optimizer = getattr(model, "optimizer", None)
        if engine_optimizer is not None:
            return engine_optimizer
        return _DeepSpeedOptimizerPlaceholder(lr, weight_decay)

    def save_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Save a DeepSpeed checkpoint.

        Args:
            model: The DeepSpeed engine.
            optimizer: The optimizer (managed by engine).
            path: Directory path for the checkpoint.
        """
        save_fn = getattr(model, "save_checkpoint", None)
        if save_fn is not None:
            save_fn(path, tag="crucible_checkpoint")
            return
        _fallback_save_checkpoint(model, optimizer, path)

    def load_checkpoint(
        self, model: Any, optimizer: Any, path: str,
    ) -> None:
        """Load a DeepSpeed checkpoint.

        Args:
            model: The DeepSpeed engine.
            optimizer: The optimizer (managed by engine).
            path: Directory path of the checkpoint.
        """
        load_fn = getattr(model, "load_checkpoint", None)
        if load_fn is not None:
            load_fn(path, tag="crucible_checkpoint")
            return
        _fallback_load_checkpoint(model, optimizer, path)

    def strategy_name(self) -> str:
        """Return the strategy name.

        Returns:
            The string 'deepspeed'.
        """
        return "deepspeed"


class _DeepSpeedOptimizerPlaceholder:
    """Placeholder when DeepSpeed manages the optimizer internally."""

    def __init__(self, lr: float, weight_decay: float) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        """No-op: DeepSpeed engine handles gradient zeroing."""

    def step(self) -> None:
        """No-op: DeepSpeed engine handles optimizer steps."""

    def state_dict(self) -> dict[str, Any]:
        """Return an empty state dict."""
        return {}


def _fallback_save_checkpoint(
    model: Any, optimizer: Any, path: str,
) -> None:
    """Fallback checkpoint save when engine API is unavailable.

    Args:
        model: The model or engine.
        optimizer: The optimizer.
        path: File path for the checkpoint.
    """
    try:
        import torch
        inner = getattr(model, "module", model)
        torch.save(
            {
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
    except ImportError:
        pass


def _fallback_load_checkpoint(
    model: Any, optimizer: Any, path: str,
) -> None:
    """Fallback checkpoint load when engine API is unavailable.

    Args:
        model: The model or engine.
        optimizer: The optimizer.
        path: File path of the checkpoint.
    """
    try:
        import torch
        checkpoint = torch.load(path, weights_only=True)
        inner = getattr(model, "module", model)
        inner.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except ImportError:
        pass
