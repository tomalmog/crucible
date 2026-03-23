"""Types for activation PCA interpretability analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationPcaOptions:
    """Configuration for an activation PCA analysis run."""

    model_path: str
    output_dir: str
    dataset_name: str
    base_model: str | None = None
    layer_index: int = -1  # -1 = last block
    max_samples: int = 500
    granularity: str = "sample"  # "sample" or "token"
    color_field: str = ""  # dataset metadata field to color by
