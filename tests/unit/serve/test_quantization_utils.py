"""Tests for quantization utilities."""

from __future__ import annotations

import pytest

from core.errors import CrucibleQloraError
from serve.quantization_utils import (
    QuantizationConfig,
    estimate_quantized_memory,
    validate_quantization_config,
)


def test_validate_valid_config() -> None:
    """Valid configs pass validation."""
    validate_quantization_config(QuantizationConfig(bits=4, quant_type="nf4"))
    validate_quantization_config(QuantizationConfig(bits=8, quant_type="fp4"))


def test_validate_invalid_bits() -> None:
    """Invalid bit width raises error."""
    with pytest.raises(CrucibleQloraError, match="must be 4 or 8"):
        validate_quantization_config(QuantizationConfig(bits=3))


def test_validate_invalid_quant_type() -> None:
    """Invalid quantization type raises error."""
    with pytest.raises(CrucibleQloraError, match="must be 'nf4' or 'fp4'"):
        validate_quantization_config(QuantizationConfig(quant_type="int8"))


def test_estimate_memory_4bit() -> None:
    """4-bit memory estimate is reasonable."""
    gb = estimate_quantized_memory(7_000_000_000, bits=4)
    assert 3.0 < gb < 5.0


def test_estimate_memory_8bit() -> None:
    """8-bit uses more memory than 4-bit."""
    gb_4 = estimate_quantized_memory(1_000_000_000, bits=4)
    gb_8 = estimate_quantized_memory(1_000_000_000, bits=8)
    assert gb_8 > gb_4
