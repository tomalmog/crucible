"""LoRA linear layer implementation.

This module provides a drop-in replacement for nn.Linear that adds
low-rank decomposition matrices A and B, computing output as
original_output + (BA * x) * (alpha / rank).
"""

from __future__ import annotations

import math
from typing import Any


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

    Args:
        torch_module: The torch module reference.
        original_linear: The nn.Linear to wrap.
        rank: Low-rank decomposition rank.
        alpha: Scaling factor.
        dropout: Dropout probability for LoRA path.

    Returns:
        A new nn.Module that wraps the original linear with LoRA.
    """
    nn = torch_module.nn
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    scaling = alpha / rank

    class LoraLinear(nn.Module):
        """Linear layer with LoRA low-rank adaptation."""

        def __init__(self) -> None:
            super().__init__()
            self.original = original_linear
            self.lora_a = nn.Parameter(
                torch_module.randn(rank, in_features) * (1.0 / math.sqrt(rank))
            )
            self.lora_b = nn.Parameter(torch_module.zeros(out_features, rank))
            self.scaling = scaling
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            for param in self.original.parameters():
                param.requires_grad = False

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
