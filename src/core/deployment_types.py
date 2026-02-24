"""Typed deployment pipeline models.

This module defines immutable data models for deployment configuration,
latency profiling, packaging, and readiness checking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for ONNX model quantization.

    Attributes:
        model_path: Path to the source ONNX model file.
        output_dir: Directory where quantized model is written.
        quantization_type: Quantization strategy (dynamic or static).
        calibration_data_path: Path to calibration data for static quant.
    """

    model_path: str
    output_dir: str
    quantization_type: str = "dynamic"
    calibration_data_path: str | None = None


@dataclass(frozen=True)
class LatencyProfile:
    """Latency profiling result for a single configuration.

    Attributes:
        batch_size: Number of samples per batch.
        sequence_length: Token count per sequence.
        device: Compute device used for profiling.
        mean_latency_ms: Mean forward-pass latency in milliseconds.
        p50_ms: Median latency in milliseconds.
        p95_ms: 95th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        throughput_tokens_per_sec: Token throughput rate.
    """

    batch_size: int
    sequence_length: int
    device: str
    mean_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_tokens_per_sec: float


@dataclass(frozen=True)
class DeploymentPackage:
    """Immutable record of a built deployment package.

    Attributes:
        package_path: Root directory of the packaged artifact.
        model_path: Path to the model file within the package.
        config_path: Path to config file within the package if present.
        tokenizer_path: Path to tokenizer file within the package if present.
        safety_report_path: Path to safety report within the package if present.
        checksum: SHA256 checksum of the model file.
    """

    package_path: str
    model_path: str
    config_path: str | None
    tokenizer_path: str | None
    safety_report_path: str | None
    checksum: str


@dataclass(frozen=True)
class DeploymentChecklist:
    """Result of a deployment readiness check.

    Attributes:
        items: Tuple of (check_name, passed) pairs.
        all_passed: True when every check passed.
    """

    items: tuple[tuple[str, bool], ...]
    all_passed: bool
