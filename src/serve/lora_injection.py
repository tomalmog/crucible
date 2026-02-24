"""LoRA adapter injection into existing models.

This module walks a model's module tree, identifies target linear layers
by name matching, and replaces them with LoRA-wrapped versions. It also
provides utilities to freeze base parameters and collect only LoRA params.
"""

from __future__ import annotations

from typing import Any

from core.errors import ForgeLoraError
from core.lora_types import LoraConfig
from core.logging_config import get_logger
from serve.lora_linear import create_lora_linear

_LOGGER = get_logger(__name__)


def inject_lora_adapters(
    torch_module: Any,
    model: Any,
    config: LoraConfig,
) -> int:
    """Replace target linear layers with LoRA-wrapped versions.

    Walks the model tree, identifies nn.Linear layers whose names
    contain any of config.target_modules, and replaces them in-place.

    Args:
        torch_module: The torch module reference.
        model: The model to inject LoRA into.
        config: LoRA configuration with rank, alpha, target modules.

    Returns:
        Number of layers replaced.

    Raises:
        ForgeLoraError: If no matching layers found.
    """
    replaced_count = 0
    for parent_name, parent_module in _named_modules_list(model):
        for child_name, child_module in _named_children_list(parent_module):
            if not _is_target_linear(torch_module, child_module, child_name, config):
                continue
            lora_layer = create_lora_linear(
                torch_module=torch_module,
                original_linear=child_module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent_module, child_name, lora_layer)
            replaced_count += 1
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            _LOGGER.info("lora_layer_injected", layer_name=full_name, rank=config.rank)

    if replaced_count == 0:
        target_list = ", ".join(config.target_modules)
        raise ForgeLoraError(
            f"No linear layers matched target modules: {target_list}. "
            "Check model architecture and target_modules config."
        )
    _LOGGER.info("lora_injection_complete", layers_replaced=replaced_count)
    return replaced_count


def freeze_base_parameters(model: Any) -> int:
    """Freeze all non-LoRA parameters in the model.

    Sets requires_grad=False for all parameters that are not LoRA A/B.

    Args:
        model: Model with injected LoRA layers.

    Returns:
        Count of frozen parameters.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            param.requires_grad = True
            continue
        param.requires_grad = False
        frozen_count += 1
    return frozen_count


def collect_lora_parameters(model: Any) -> list[Any]:
    """Collect only LoRA-trainable parameters.

    Returns:
        List of parameters with requires_grad=True (LoRA A/B matrices).
    """
    return [p for p in model.parameters() if p.requires_grad]


def _is_target_linear(
    torch_module: Any,
    module: Any,
    name: str,
    config: LoraConfig,
) -> bool:
    """Check if a module is a target nn.Linear for LoRA injection."""
    if not isinstance(module, torch_module.nn.Linear):
        return False
    return any(target in name for target in config.target_modules)


def _named_modules_list(model: Any) -> list[tuple[str, Any]]:
    """Return named modules as a stable list to allow in-place mutation."""
    return list(model.named_modules())


def _named_children_list(module: Any) -> list[tuple[str, Any]]:
    """Return named children as a stable list to allow in-place mutation."""
    return list(module.named_children())
