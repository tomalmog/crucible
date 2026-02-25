"""LoRA linear layer implementation.

This module provides a drop-in replacement for nn.Linear that adds
low-rank decomposition matrices A and B, computing output as
original_output + (BA * x) * (alpha / rank).
"""

from __future__ import annotations

import math
from typing import Any


def _resolve_features(original_linear: Any) -> tuple[int, int]:
    """Extract in/out features from nn.Linear or transformers Conv1D."""
    if hasattr(original_linear, "in_features"):
        return original_linear.in_features, original_linear.out_features
    if hasattr(original_linear, "nf"):
        # transformers Conv1D: weight is (in_features, nf), nf = out_features
        return original_linear.weight.shape[0], original_linear.nf
    w = original_linear.weight
    return w.shape[1], w.shape[0]


def create_lora_linear(
    torch_module: Any,
    original_linear: Any,
    rank: int,
    alpha: float,
    dropout: float,
) -> Any:
    """Create a LoRA-wrapped linear layer.

    Stores the original linear weights (frozen) plus trainable A/B matrices.
    Forward computes: original(x) + dropout(x) @ A^T @ B^T * scaling.

    Supports both nn.Linear and transformers Conv1D layers.

    Args:
        torch_module: The torch module reference.
        original_linear: The nn.Linear or Conv1D to wrap.
        rank: Low-rank decomposition rank.
        alpha: Scaling factor.
        dropout: Dropout probability for LoRA path.

    Returns:
        A new nn.Module that wraps the original linear with LoRA.
    """
    nn = torch_module.nn
    in_features, out_features = _resolve_features(original_linear)
    scaling = alpha / rank

    class LoraLinear(nn.Module):
        """Linear layer with LoRA low-rank adaptation."""

        def __init__(self) -> None:
            super().__init__()
            self.original = original_linear
            device = original_linear.weight.device
            dtype = original_linear.weight.dtype
            self.lora_a = nn.Parameter(
                torch_module.randn(rank, in_features, device=device, dtype=dtype)
                * (1.0 / math.sqrt(rank))
            )
            self.lora_b = nn.Parameter(
                torch_module.zeros(out_features, rank, device=device, dtype=dtype)
            )
            self.scaling = scaling
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            for param in self.original.parameters():
                param.requires_grad = False

        @property
        def weight(self) -> Any:
            """Expose original weight for modules that access it directly."""
            return self.original.weight

        @property
        def bias(self) -> Any:
            """Expose original bias for modules that access it directly."""
            return self.original.bias

        def forward(self, x: Any) -> Any:
            """Compute original output plus LoRA delta."""
            base_output = self.original(x)
            lora_input = self.lora_dropout(x)
            lora_output = (lora_input @ self.lora_a.T) @ self.lora_b.T
            return base_output + lora_output * self.scaling

    return LoraLinear()


def is_lora_linear(module: Any) -> bool:
    """Check whether a module is a LoRA-wrapped linear layer.

    Returns:
        True if the module has lora_a and lora_b parameters.
    """
    return hasattr(module, "lora_a") and hasattr(module, "lora_b")
