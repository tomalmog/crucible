"""ONNX export option types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OnnxExportOptions:
    """Configuration for exporting a model to ONNX format."""

    model_path: str
    output_dir: str
    opset_version: int = 17
