"""Unit tests for deployment type definitions."""

from __future__ import annotations

import pytest

from core.deployment_types import (
    DeploymentChecklist,
    DeploymentPackage,
    LatencyProfile,
    QuantizationConfig,
)


def test_quantization_config_is_frozen() -> None:
    """QuantizationConfig should be immutable after construction."""
    config = QuantizationConfig(
        model_path="/tmp/model.onnx",
        output_dir="/tmp/out",
    )
    with pytest.raises(AttributeError):
        config.model_path = "/changed"  # type: ignore[misc]


def test_latency_profile_fields_present() -> None:
    """LatencyProfile should expose all expected fields."""
    profile = LatencyProfile(
        batch_size=4,
        sequence_length=128,
        device="cpu",
        mean_latency_ms=10.5,
        p50_ms=9.8,
        p95_ms=15.2,
        p99_ms=18.1,
        throughput_tokens_per_sec=48762.0,
    )
    assert profile.batch_size == 4
    assert profile.sequence_length == 128
    assert profile.device == "cpu"
    assert profile.mean_latency_ms == 10.5
    assert profile.p50_ms == 9.8
    assert profile.p95_ms == 15.2
    assert profile.p99_ms == 18.1
    assert profile.throughput_tokens_per_sec == 48762.0


def test_deployment_checklist_all_passed() -> None:
    """DeploymentChecklist.all_passed should reflect item results."""
    checklist_pass = DeploymentChecklist(
        items=(("model_exists", True), ("config_exists", True)),
        all_passed=True,
    )
    assert checklist_pass.all_passed is True

    checklist_fail = DeploymentChecklist(
        items=(("model_exists", True), ("config_exists", False)),
        all_passed=False,
    )
    assert checklist_fail.all_passed is False
