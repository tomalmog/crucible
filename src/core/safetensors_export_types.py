"""SafeTensors export option types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafeTensorsExportOptions:
    """Configuration for exporting a model to SafeTensors format."""

    model_path: str
    output_dir: str
