"""Activation steering options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SteerComputeOptions:
    model_path: str
    output_dir: str
    positive_text: str = ""
    negative_text: str = ""
    positive_dataset: str = ""
    negative_dataset: str = ""
    base_model: str | None = None
    layer_index: int = -1
    max_samples: int = 100


@dataclass(frozen=True)
class SteerApplyOptions:
    model_path: str
    output_dir: str
    steering_vector_path: str
    input_text: str
    coefficient: float = 1.0
    max_new_tokens: int = 50
    base_model: str | None = None
