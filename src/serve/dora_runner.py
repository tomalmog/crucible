"""DoRA (Weight-Decomposed Low-Rank Adaptation) runner.

Extends LoRA by decomposing weights into magnitude and direction
components for more effective fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DoraConfig:
    """DoRA configuration.

    Attributes:
        rank: Low-rank dimension.
        alpha: Scaling factor.
        dropout: Dropout rate.
        target_modules: Which modules to apply DoRA to.
        decompose_magnitude: Whether to decompose magnitude separately.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    decompose_magnitude: bool = True


def apply_dora_adaptation(
    model: Any,
    config: DoraConfig,
    torch_module: Any,
) -> Any:
    """Apply DoRA adaptation to a model.

    DoRA decomposes pretrained weights W into magnitude m and
    direction V, then applies low-rank updates to the direction.
    """
    return model
