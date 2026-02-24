"""Hyperparameter sweep result analysis.

This module provides ranking, filtering, and formatting utilities
for sweep trial results used by the CLI and sweep runner.
"""

from __future__ import annotations

from core.sweep_types import SweepResult, SweepTrialResult


def rank_trials(
    trials: list[SweepTrialResult],
    minimize: bool,
) -> list[SweepTrialResult]:
    """Sort trials by metric value.

    Args:
        trials: List of completed trial results.
        minimize: True to sort ascending (lower is better).

    Returns:
        Sorted list of trial results.
    """
    return sorted(
        trials,
        key=lambda t: t.metric_value,
        reverse=not minimize,
    )


def find_best_trial(
    ranked_trials: list[SweepTrialResult],
) -> SweepTrialResult:
    """Return the best trial from a ranked list.

    Args:
        ranked_trials: Trials sorted by metric (best first).

    Returns:
        Best trial result.
    """
    return ranked_trials[0]


def top_k_trials(
    result: SweepResult,
    k: int,
) -> tuple[SweepTrialResult, ...]:
    """Return the top-k trials from a sweep result.

    Args:
        result: Full sweep result.
        k: Number of top trials to return.

    Returns:
        Tuple of up to k best trial results.
    """
    ranked = rank_trials(list(result.trials), minimize=True)
    best_is_min = result.best_metric_value == ranked[0].metric_value
    if not best_is_min:
        ranked = rank_trials(list(result.trials), minimize=False)
    return tuple(ranked[:k])


def format_sweep_report(result: SweepResult) -> str:
    """Format a human-readable sweep summary report.

    Args:
        result: Full sweep result with all trials.

    Returns:
        Multi-line summary string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Hyperparameter Sweep Report")
    lines.append("=" * 60)
    lines.append(f"Total trials: {len(result.trials)}")
    lines.append(f"Best trial:   #{result.best_trial_id}")
    lines.append(f"Best metric:  {result.best_metric_value:.6f}")
    lines.append("")
    lines.append("Best parameters:")
    for name, value in sorted(result.best_parameters.items()):
        lines.append(f"  {name}: {value}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("All trials (sorted by metric):")
    lines.append("-" * 60)
    sorted_trials = sorted(
        result.trials,
        key=lambda t: t.metric_value,
    )
    for trial in sorted_trials:
        param_str = _format_params_compact(trial.parameters)
        lines.append(
            f"  Trial #{trial.trial_id:4d}  "
            f"metric={trial.metric_value:.6f}  "
            f"{param_str}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def _format_params_compact(params: dict[str, float]) -> str:
    """Format parameter dict as compact key=value string.

    Args:
        params: Parameter name-to-value mapping.

    Returns:
        Compact string representation.
    """
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return " ".join(parts)
