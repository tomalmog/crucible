"""Unit tests for training cost estimator."""

from __future__ import annotations

from compute.cost_estimator import (
    CostEstimate,
    estimate_training_cost,
    format_cost_estimate,
)


def test_estimate_returns_positive_hours() -> None:
    """estimate_training_cost should return positive hour values."""
    estimate = estimate_training_cost(
        epochs=10,
        batch_size=32,
        dataset_size=10000,
        gpu_count=1,
    )
    assert estimate.estimated_hours > 0
    assert estimate.estimated_gpu_hours > 0
    assert estimate.confidence == "low"


def test_estimate_scales_with_gpu_count() -> None:
    """Doubling GPU count should roughly halve estimated hours."""
    single = estimate_training_cost(
        epochs=5, batch_size=16, dataset_size=5000, gpu_count=1,
    )
    double = estimate_training_cost(
        epochs=5, batch_size=16, dataset_size=5000, gpu_count=2,
    )
    assert double.estimated_hours < single.estimated_hours


def test_format_cost_estimate_returns_tuple() -> None:
    """format_cost_estimate should return a tuple of human-readable strings."""
    estimate = CostEstimate(
        estimated_hours=1.5,
        estimated_gpu_hours=3.0,
        confidence="low",
    )
    lines = format_cost_estimate(estimate)
    assert isinstance(lines, tuple)
    assert len(lines) == 3
    assert "1.5000" in lines[0]
    assert "3.0000" in lines[1]
    assert "low" in lines[2]
