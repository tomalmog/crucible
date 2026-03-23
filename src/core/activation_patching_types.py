"""Types for activation patching interpretability analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationPatchingOptions:
    """Configuration for an activation patching analysis run."""

    model_path: str
    output_dir: str
    clean_text: str
    corrupted_text: str
    target_token_index: int = -1  # -1 = last token
    base_model: str | None = None
    metric: str = "logit_diff"  # "logit_diff" or "prob"
