"""LoRA adapter save/load and merge operations.

This module persists trained LoRA adapter weights, loads them back
for inference, and merges adapter parameters into the base model
for deployment without LoRA overhead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import ForgeLoraError
from core.lora_types import LoraAdapterInfo, LoraConfig
from core.logging_config import get_logger

_LOGGER = get_logger(__name__)

_ADAPTER_WEIGHTS_FILE = "lora_adapter.pt"
_ADAPTER_CONFIG_FILE = "lora_adapter_config.json"


def save_lora_adapter(
    torch_module: Any,
    model: Any,
    output_dir: Path,
    config: LoraConfig,
) -> LoraAdapterInfo:
    """Save trained LoRA adapter weights and config to disk.

    Extracts only LoRA parameters (lora_a, lora_b) from the model
    and persists them alongside adapter configuration.

    Returns:
        Adapter info with saved path and configuration.
    """
    adapter_state = _extract_lora_state_dict(model)
    if not adapter_state:
        raise ForgeLoraError("No LoRA parameters found in model to save.")
    weights_path = output_dir / _ADAPTER_WEIGHTS_FILE
    torch_module.save(adapter_state, str(weights_path))
    config_payload = {
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "target_modules": list(config.target_modules),
    }
    config_path = output_dir / _ADAPTER_CONFIG_FILE
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    _LOGGER.info(
        "lora_adapter_saved",
        weights_path=str(weights_path),
        parameter_count=len(adapter_state),
    )
    return LoraAdapterInfo(
        adapter_path=str(weights_path),
        rank=config.rank,
        alpha=config.alpha,
        target_modules=config.target_modules,
    )


def load_lora_adapter(
    torch_module: Any,
    model: Any,
    adapter_path: str,
    device: Any,
) -> None:
    """Load LoRA adapter weights into an injected model.

    The model must already have LoRA layers injected before loading.

    Raises:
        ForgeLoraError: If adapter file is missing or incompatible.
    """
    path = Path(adapter_path)
    if not path.exists():
        raise ForgeLoraError(f"Adapter weights not found at {adapter_path}.")
    adapter_state = torch_module.load(str(path), map_location=device)
    model_state = model.state_dict()
    matched = 0
    for key, value in adapter_state.items():
        if key in model_state:
            model_state[key] = value
            matched += 1
    if matched == 0:
        raise ForgeLoraError(
            "No adapter keys matched model state. Ensure LoRA layers are injected."
        )
    model.load_state_dict(model_state)
    _LOGGER.info("lora_adapter_loaded", matched_keys=matched)


def merge_lora_into_base(
    torch_module: Any,
    model: Any,
    output_path: str,
) -> str:
    """Merge LoRA weights into base model and save merged weights.

    For each LoRA layer, computes: W_merged = W_original + B @ A * scaling
    then replaces the LoRA layer with a standard nn.Linear.

    Returns:
        Path to saved merged model weights.
    """
    merged_count = _merge_lora_layers(torch_module, model)
    if merged_count == 0:
        raise ForgeLoraError("No LoRA layers found in model to merge.")
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(model.state_dict(), str(resolved_path))
    _LOGGER.info(
        "lora_merged_and_saved",
        merged_layers=merged_count,
        output_path=str(resolved_path),
    )
    return str(resolved_path)


def _extract_lora_state_dict(model: Any) -> dict[str, Any]:
    """Extract only LoRA parameters from model state dict."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if "lora_a" in key or "lora_b" in key
    }


def _merge_lora_layers(torch_module: Any, model: Any) -> int:
    """Merge LoRA deltas into original weights and replace LoRA modules.

    After merging, each LoraLinear module is replaced with its inner
    ``original`` nn.Linear (now containing merged weights) so that
    ``model.state_dict()`` produces clean keys (``weight``, ``bias``)
    instead of LoRA-structured keys (``original.weight``, ``lora_a``, etc.).
    """
    # Collect (parent, attr_name, lora_module) before mutating the tree
    replacements: list[tuple[Any, str, Any]] = []
    for name, module in list(model.named_modules()):
        if not (hasattr(module, "lora_a") and hasattr(module, "original")):
            continue
        _merge_single_layer(torch_module, module)
        # Find the parent module so we can swap out the LoraLinear
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]
        replacements.append((parent, attr, module.original))

    for parent, attr, merged_linear in replacements:
        setattr(parent, attr, merged_linear)

    return len(replacements)


def _merge_single_layer(torch_module: Any, lora_module: Any) -> None:
    """Merge one LoRA layer's delta into its original weight."""
    with torch_module.no_grad():
        delta = (lora_module.lora_b @ lora_module.lora_a) * lora_module.scaling
        lora_module.original.weight.add_(delta)
