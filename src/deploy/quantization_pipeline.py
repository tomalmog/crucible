"""ONNX model quantization pipeline.

This module provides dynamic and static quantization of ONNX models
using the optional onnxruntime dependency. All external imports are
guarded so the module loads cleanly when onnxruntime is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.deployment_types import QuantizationConfig
from core.errors import CrucibleDeployError, CrucibleDependencyError

_VALID_QUANTIZATION_TYPES = ("dynamic", "static")


def validate_quantization_config(config: QuantizationConfig) -> None:
    """Validate a quantization configuration before execution.

    Args:
        config: Quantization configuration to validate.

    Raises:
        CrucibleDeployError: If model path does not exist or type is invalid.
    """
    if not os.path.isfile(config.model_path):
        raise CrucibleDeployError(
            f"Model file not found: {config.model_path}"
        )
    if not config.model_path.endswith(".onnx"):
        raise CrucibleDeployError(
            f"Quantization requires an ONNX model, got '{config.model_path}'. "
            "Export your PyTorch model to ONNX first with: crucible deploy package --format onnx"
        )
    if config.quantization_type not in _VALID_QUANTIZATION_TYPES:
        raise CrucibleDeployError(
            f"Invalid quantization type '{config.quantization_type}'. "
            f"Must be one of: {_VALID_QUANTIZATION_TYPES}"
        )
    if (
        config.quantization_type == "static"
        and config.calibration_data_path is not None
        and not os.path.exists(config.calibration_data_path)
    ):
        raise CrucibleDeployError(
            f"Calibration data not found: {config.calibration_data_path}"
        )


def run_quantization(config: QuantizationConfig) -> str:
    """Run ONNX quantization on a model.

    Args:
        config: Quantization configuration.

    Returns:
        Path to the quantized model file.

    Raises:
        CrucibleDeployError: If validation fails or quantization errors.
        CrucibleDependencyError: If onnxruntime is not installed.
    """
    validate_quantization_config(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model_name = Path(config.model_path).stem
    output_path = str(
        Path(config.output_dir) / f"{model_name}_quantized.onnx"
    )

    if config.quantization_type == "dynamic":
        return _run_dynamic_quantization(
            config.model_path, output_path,
        )
    return _run_static_quantization(
        config.model_path,
        output_path,
        config.calibration_data_path,
    )


def _run_dynamic_quantization(
    model_path: str, output_path: str,
) -> str:
    """Run dynamic quantization using onnxruntime.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for the quantized output model.

    Returns:
        Path to the quantized model file.

    Raises:
        CrucibleDependencyError: If onnxruntime is not available.
        CrucibleDeployError: If quantization fails.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantType,
            quantize_dynamic,
        )
    except ImportError as exc:
        raise CrucibleDependencyError(
            "onnxruntime is required for quantization. "
            "Install with: pip install onnxruntime"
        ) from exc

    try:
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
    except Exception as exc:
        raise CrucibleDeployError(
            f"Dynamic quantization failed: {exc}"
        ) from exc

    return output_path


def _run_static_quantization(
    model_path: str,
    output_path: str,
    calibration_path: str | None,
) -> str:
    """Run static quantization using onnxruntime.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for the quantized output model.
        calibration_path: Optional path to calibration data.

    Returns:
        Path to the quantized model file.

    Raises:
        CrucibleDependencyError: If onnxruntime is not available.
        CrucibleDeployError: If quantization fails.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise CrucibleDependencyError(
            "onnxruntime is required for quantization. "
            "Install with: pip install onnxruntime"
        ) from exc

    try:
        quantize_static(
            model_path,
            output_path,
            calibration_data_reader=None,
            quant_format=QuantType.QUInt8,
        )
    except Exception as exc:
        raise CrucibleDeployError(
            f"Static quantization failed: {exc}"
        ) from exc

    return output_path
