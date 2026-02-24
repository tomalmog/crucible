"""Gradient checkpointing for memory-efficient training.

Wraps qualifying compound model layers so intermediate activations are
recomputed during backward instead of stored in memory, reducing peak
GPU memory usage at the cost of additional compute.
"""

from __future__ import annotations

from typing import Any


def apply_gradient_checkpointing(torch_module: Any, model: Any) -> None:
    """Enable gradient checkpointing on compound child layers.

    Walks the model's immediate children and wraps any compound module
    (one that itself contains sub-modules) so its forward pass is
    executed through ``torch.utils.checkpoint.checkpoint``.

    Args:
        torch_module: The imported ``torch`` module.
        model: A ``torch.nn.Module`` whose children may be wrapped.
    """
    checkpoint_module = _resolve_checkpoint_module(torch_module)
    if checkpoint_module is None:
        return
    layers = _find_checkpointable_layers(model)
    for layer in layers:
        _wrap_layer_forward(layer, checkpoint_module)


def _resolve_checkpoint_module(torch_module: Any) -> Any | None:
    """Return ``torch.utils.checkpoint`` or *None* if unavailable."""
    utils = getattr(torch_module, "utils", None)
    if utils is None:
        return None
    return getattr(utils, "checkpoint", None)


def _find_checkpointable_layers(model: Any) -> list[Any]:
    """Return immediate children that themselves contain sub-modules."""
    result: list[Any] = []
    for child in model.children():
        if _has_sub_modules(child):
            result.append(child)
    return result


def _has_sub_modules(module: Any) -> bool:
    """Return *True* if *module* has at least one child module."""
    return len(list(module.children())) > 0


def _wrap_layer_forward(layer: Any, checkpoint_module: Any) -> None:
    """Replace *layer.forward* with a checkpointed version."""
    original_forward = layer.forward
    checkpoint_fn = checkpoint_module.checkpoint

    def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
        return checkpoint_fn(
            original_forward, *args, use_reentrant=False, **kwargs
        )

    layer.forward = checkpointed_forward  # type: ignore[method-assign]
