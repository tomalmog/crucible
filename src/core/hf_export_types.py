"""HuggingFace export option types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfExportOptions:
    """Configuration for exporting a model to HuggingFace-compatible format."""

    model_path: str
    output_dir: str
