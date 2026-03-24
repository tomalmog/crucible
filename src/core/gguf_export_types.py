"""GGUF export option types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GgufQuantType = Literal["F32", "F16", "Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M"]


@dataclass(frozen=True)
class GgufExportOptions:
    """Configuration for exporting a model to GGUF format."""

    model_path: str
    output_dir: str
    quant_type: GgufQuantType = "F16"
