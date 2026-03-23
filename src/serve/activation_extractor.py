"""Shared activation extraction for interpretability analyses.

Provides a context-manager that attaches PyTorch forward hooks to named
layers and captures their outputs. Also includes helpers for discovering
transformer block names and extracting the unembedding matrix from both
Crucible and HuggingFace models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerActivations:
    """Container for captured activations from a single layer."""

    layer_name: str
    activations: Any  # torch.Tensor [batch, seq_len, hidden_dim]


class ActivationExtractor:
    """Context manager for capturing layer outputs via forward hooks."""

    def __init__(self, model: Any, layer_names: list[str]) -> None:
        self._model = model
        self._layer_names = layer_names
        self._activations: dict[str, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationExtractor":
        for name in self._layer_names:
            module = _get_module_by_name(self._model, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_activations(self) -> dict[str, Any]:
        """Return captured activations keyed by layer name."""
        return dict(self._activations)

    def clear(self) -> None:
        """Clear captured activations for next forward pass."""
        self._activations.clear()

    def _make_hook(self, name: str) -> Any:
        def hook(_module: Any, _input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                output = output[0]
            self._activations[name] = output.detach()

        return hook


def _get_module_by_name(model: Any, name: str) -> Any:
    """Resolve a dotted module name like 'model.layers.0' to a module."""
    parts = name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def discover_transformer_layers(model: Any) -> list[str]:
    """Return hookable transformer block names for any model."""
    patterns = [
        ("blocks",),                    # Crucible DefaultCausalModel
        ("model", "layers"),            # HF Llama, Mistral, Phi
        ("transformer", "h"),           # HF GPT2, GPT-J
        ("transformer", "blocks"),      # HF MPT
        ("gpt_neox", "layers"),         # HF Pythia, GPT-NeoX
    ]
    for path in patterns:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            prefix = ".".join(path)
            return [f"{prefix}.{i}" for i in range(len(obj))]
    raise ValueError(
        "Could not discover transformer layers. "
        "Ensure the model has a standard block structure."
    )


def get_unembedding(model: Any) -> Any:
    """Return the final projection (hidden -> vocab logits)."""
    candidates = ["output", "lm_head"]
    for attr in candidates:
        mod = model
        for part in attr.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is not None:
            return mod
    # HF logits wrapper: unwrap .model.lm_head
    inner = getattr(model, "model", None)
    if inner is not None:
        head = getattr(inner, "lm_head", None)
        if head is not None:
            return head
    raise ValueError(
        "Could not find unembedding layer. "
        "Expected 'output' (Crucible) or 'lm_head' (HuggingFace)."
    )
