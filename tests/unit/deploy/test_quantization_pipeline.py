"""Unit tests for ONNX quantization pipeline."""

from __future__ import annotations

import os
import tempfile

import pytest

from core.deployment_types import QuantizationConfig
from core.errors import ForgeDeployError
from deploy.quantization_pipeline import (
    run_quantization,
    validate_quantization_config,
)


def test_validate_config_rejects_missing_model() -> None:
    """Validation should raise ForgeDeployError for a missing model."""
    config = QuantizationConfig(
        model_path="/nonexistent/model.onnx",
        output_dir="/tmp/out",
        quantization_type="dynamic",
    )
    with pytest.raises(ForgeDeployError, match="Model file not found"):
        validate_quantization_config(config)


def test_validate_config_rejects_invalid_type() -> None:
    """Validation should raise ForgeDeployError for invalid quant type."""
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        config = QuantizationConfig(
            model_path=tmp.name,
            output_dir="/tmp/out",
            quantization_type="invalid_type",
        )
        with pytest.raises(ForgeDeployError, match="Invalid quantization type"):
            validate_quantization_config(config)


def test_validate_config_accepts_dynamic_type() -> None:
    """Validation should pass for a valid dynamic config."""
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        config = QuantizationConfig(
            model_path=tmp.name,
            output_dir="/tmp/out",
            quantization_type="dynamic",
        )
        # Should not raise
        validate_quantization_config(config)
