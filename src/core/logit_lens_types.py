"""Types for logit lens interpretability analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogitLensOptions:
    """Configuration for a logit lens analysis run."""

    model_path: str
    output_dir: str
    input_text: str
    base_model: str | None = None
    top_k: int = 5
    layer_indices: str = ""  # comma-separated, empty = all
