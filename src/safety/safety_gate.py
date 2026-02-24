"""Pre-deploy safety gate.

Evaluates model outputs against a toxicity threshold
to determine if a model is safe for deployment.
"""

from __future__ import annotations

from core.errors import ForgeSafetyError
from core.safety_types import SafetyEvalConfig, SafetyGateResult, SafetyReport
from safety.toxicity_scorer import score_batch_toxicity


def run_safety_gate(
    config: SafetyEvalConfig,
    texts: list[str],
) -> SafetyGateResult:
    """Run a safety gate check on a set of texts.

    Scores all texts for toxicity and checks whether the
    mean score falls below the configured threshold.

    Args:
        config: Safety evaluation configuration.
        texts: Texts to evaluate.

    Returns:
        Safety gate result with pass/fail status.

    Raises:
        ForgeSafetyError: If no texts are provided.
    """
    if not texts:
        raise ForgeSafetyError("No texts provided for safety gate.")

    threshold = config.toxicity_threshold
    scores = score_batch_toxicity(texts, threshold=threshold)
    score_values = [s.score for s in scores]

    mean_score = sum(score_values) / len(score_values)
    max_score = max(score_values)
    flagged_count = sum(1 for s in scores if s.flagged)

    report = SafetyReport(
        total_samples=len(texts),
        flagged_count=flagged_count,
        mean_score=mean_score,
        max_score=max_score,
        scores=tuple(scores),
    )

    passed = mean_score < threshold

    return SafetyGateResult(
        passed=passed,
        report=report,
        threshold=threshold,
        gate_name="toxicity-gate",
    )


def format_gate_result(
    result: SafetyGateResult,
) -> tuple[str, ...]:
    """Format a gate result as human-readable lines.

    Args:
        result: The safety gate result to format.

    Returns:
        Tuple of formatted summary lines.
    """
    status = "PASSED" if result.passed else "FAILED"
    lines: list[str] = [
        f"=== Safety Gate: {result.gate_name} ===",
        f"Status: {status}",
        f"Threshold: {result.threshold:.4f}",
        f"Mean toxicity: {result.report.mean_score:.4f}",
        f"Max toxicity: {result.report.max_score:.4f}",
        f"Flagged: {result.report.flagged_count}"
        f"/{result.report.total_samples}",
    ]
    return tuple(lines)
