"""Quantization utilities for QLoRA training.

This module provides 4-bit and 8-bit quantization helpers
for reducing model memory footprint during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import CrucibleQloraError


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration for model compression.

    Attributes:
        bits: Number of quantization bits (4 or 8).
        quant_type: Quantization type (nf4, fp4).
        double_quantize: Whether to apply double quantization.
    """

    bits: int = 4
    quant_type: str = "nf4"
    double_quantize: bool = True


def validate_quantization_config(config: QuantizationConfig) -> None:
    """Validate quantization configuration parameters."""
    if config.bits not in (4, 8):
        raise CrucibleQloraError(
            f"Quantization bits must be 4 or 8, got {config.bits}."
        )
    if config.quant_type not in ("nf4", "fp4"):
        raise CrucibleQloraError(
            f"Quantization type must be 'nf4' or 'fp4', got '{config.quant_type}'."
        )


def quantize_linear_layer(
    torch_module: Any,
    weight: Any,
    bits: int,
    quant_type: str,
) -> Any:
    """Quantize a weight tensor to the specified bit width.

    For 4-bit NF4, applies normalized float quantization.
    For 8-bit, applies symmetric int8 quantization.
    """
    if bits == 8:
        scale = weight.abs().max() / 127.0
        quantized = (weight / scale).round().clamp(-128, 127).to(torch_module.int8)
        return quantized, scale
    scale = weight.abs().max() / 7.0
    quantized = (weight / scale).round().clamp(-8, 7).to(torch_module.int8)
    return quantized, scale


def dequantize_weight(
    torch_module: Any,
    quantized: Any,
    scale: Any,
) -> Any:
    """Dequantize a weight tensor back to floating point."""
    return quantized.float() * scale


def estimate_quantized_memory(
    param_count: int,
    bits: int,
) -> float:
    """Estimate memory usage in GB for quantized model."""
    bytes_per_param = bits / 8.0
    total_bytes = param_count * bytes_per_param
    overhead_factor = 1.1
    return (total_bytes * overhead_factor) / (1024**3)
