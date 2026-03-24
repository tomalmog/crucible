"""Linear probe options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearProbeOptions:
    model_path: str
    output_dir: str
    dataset_name: str
    label_field: str
    base_model: str | None = None
    layer_index: int = -1  # -1 = last, -2 = all layers
    max_samples: int = 500
    epochs: int = 10
    learning_rate: float = 1e-3
