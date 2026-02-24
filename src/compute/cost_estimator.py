"""Training cost estimation heuristics.

This module provides simple heuristic-based estimates for training
runtime and GPU-hours based on dataset size and hardware configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    """Estimated resource consumption for a training job.

    Attributes:
        estimated_hours: Projected wall-clock hours.
        estimated_gpu_hours: Projected GPU-hours (hours * gpu_count).
        confidence: Estimate reliability ("low", "medium", "high").
    """

    estimated_hours: float
    estimated_gpu_hours: float
    confidence: str


def estimate_training_cost(
    epochs: int,
    batch_size: int,
    dataset_size: int,
    gpu_count: int,
) -> CostEstimate:
    """Estimate training cost from dataset and hardware parameters.

    Uses a simple throughput heuristic: samples processed per second
    scales linearly with batch_size and gpu_count.

    Args:
        epochs: Number of training epochs.
        batch_size: Samples per training step.
        dataset_size: Total number of samples in the dataset.
        gpu_count: Number of GPUs available.

    Returns:
        A CostEstimate with projected hours and confidence.
    """
    effective_gpu = max(gpu_count, 1)
    effective_batch = max(batch_size, 1)
    total_samples = epochs * dataset_size
    throughput_per_second = effective_batch * effective_gpu
    total_seconds = total_samples / throughput_per_second
    hours = total_seconds / 3600.0
    gpu_hours = hours * effective_gpu
    return CostEstimate(
        estimated_hours=round(hours, 4),
        estimated_gpu_hours=round(gpu_hours, 4),
        confidence="low",
    )


def format_cost_estimate(estimate: CostEstimate) -> tuple[str, ...]:
    """Format a cost estimate into human-readable summary lines.

    Args:
        estimate: The cost estimate to format.

    Returns:
        Tuple of formatted summary strings.
    """
    return (
        f"Estimated wall-clock time: {estimate.estimated_hours:.4f} hours",
        f"Estimated GPU-hours: {estimate.estimated_gpu_hours:.4f}",
        f"Confidence: {estimate.confidence}",
    )
