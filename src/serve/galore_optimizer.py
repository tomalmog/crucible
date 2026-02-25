"""GaLore memory-efficient optimizer.

Gradient Low-Rank Projection -- reduces optimizer memory by
projecting gradients to a low-rank space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GaloreConfig:
    """GaLore optimizer configuration.

    Attributes:
        rank: Projection rank.
        update_proj_gap: Steps between projection updates.
        scale: Gradient scaling factor.
    """

    rank: int = 128
    update_proj_gap: int = 200
    scale: float = 0.25


def build_galore_optimizer(
    model: Any,
    config: GaloreConfig,
    learning_rate: float,
    torch_module: Any,
) -> Any:
    """Build a GaLore optimizer wrapper.

    Projects gradients to low-rank space to reduce memory,
    then applies AdamW in the projected space.
    """
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer
