"""Drift detection for domain adaptation workflows.

This module tracks perplexity on reference data to detect catastrophic
forgetting during continued pretraining. It provides utilities to compute
perplexity and compare against a baseline threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DriftCheckResult:
    """Result of one drift check against reference data.

    Attributes:
        epoch: One-based epoch index when the check was performed.
        perplexity: Measured perplexity on reference data.
        baseline_perplexity: Perplexity measured before training began.
        drift_detected: True if perplexity exceeded the threshold.
    """

    epoch: int
    perplexity: float
    baseline_perplexity: float
    drift_detected: bool


def compute_perplexity(
    torch_module: Any,
    model: Any,
    sequences: list[list[int]],
    device: Any,
) -> float:
    """Compute perplexity of a model on a set of token sequences.

    Args:
        torch_module: The torch module reference.
        model: The model to evaluate.
        sequences: List of integer token sequences.
        device: Torch device for computation.

    Returns:
        Perplexity as exp(average cross-entropy loss).
    """
    if not sequences:
        return float("inf")
    loss_fn = torch_module.nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch_module.no_grad():
        for seq in sequences:
            if len(seq) < 2:
                continue
            input_ids = torch_module.tensor(
                [seq[:-1]], dtype=torch_module.long, device=device,
            )
            target_ids = torch_module.tensor(
                [seq[1:]], dtype=torch_module.long, device=device,
            )
            logits = model(input_ids)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )
            total_loss += loss.item() * (len(seq) - 1)
            total_count += len(seq) - 1
    model.train()
    if total_count == 0:
        return float("inf")
    avg_loss = total_loss / total_count
    return math.exp(avg_loss)


def check_drift(
    baseline_perplexity: float,
    current_perplexity: float,
    max_increase_ratio: float,
) -> bool:
    """Check whether perplexity drift exceeds the allowed threshold.

    Args:
        baseline_perplexity: Perplexity measured before adaptation.
        current_perplexity: Perplexity measured at current epoch.
        max_increase_ratio: Maximum allowed ratio of current/baseline.

    Returns:
        True if drift is detected (perplexity increased beyond threshold).
    """
    if baseline_perplexity <= 0.0:
        return False
    return current_perplexity > baseline_perplexity * max_increase_ratio
