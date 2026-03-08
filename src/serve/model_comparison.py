"""Training run comparison logic.

This module compares two training run records by examining their metadata,
configuration hashes, datasets, and lifecycle states. It produces a structured
report of differences between runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from serve.training_run_types import TrainingRunRecord


@dataclass(frozen=True)
class RunComparisonReport:
    """Structured comparison result between two training runs."""

    run_id_a: str
    run_id_b: str
    config_delta: dict[str, tuple[object, object]]
    state_a: str
    state_b: str
    dataset_a: str
    dataset_b: str
    created_at_a: str
    created_at_b: str


def compare_training_runs(
    run_a: TrainingRunRecord,
    run_b: TrainingRunRecord,
) -> RunComparisonReport:
    """Compare two training run records and produce a delta report.

    Args:
        run_a: First training run record.
        run_b: Second training run record.

    Returns:
        Structured comparison report.
    """
    config_a = _extract_config_fields(run_a)
    config_b = _extract_config_fields(run_b)
    config_delta = _compute_config_delta(config_a, config_b)
    return RunComparisonReport(
        run_id_a=run_a.run_id,
        run_id_b=run_b.run_id,
        config_delta=config_delta,
        state_a=run_a.state,
        state_b=run_b.state,
        dataset_a=run_a.dataset_name,
        dataset_b=run_b.dataset_name,
        created_at_a=run_a.created_at,
        created_at_b=run_b.created_at,
    )


def _extract_config_fields(record: TrainingRunRecord) -> dict[str, object]:
    """Extract comparable configuration fields from a run record.

    Args:
        record: Training run record.

    Returns:
        Dictionary of configuration-relevant fields.
    """
    return {
        "dataset_name": record.dataset_name,
        "output_dir": record.output_dir,
        "parent_model_path": record.parent_model_path,
        "config_hash": record.config_hash,
    }


def _compute_config_delta(
    config_a: dict[str, object],
    config_b: dict[str, object],
) -> dict[str, tuple[object, object]]:
    """Find differing configuration keys between two config dicts.

    Args:
        config_a: Configuration from run A.
        config_b: Configuration from run B.

    Returns:
        Mapping of differing keys to (value_a, value_b) tuples.
    """
    all_keys = sorted(set(config_a.keys()) | set(config_b.keys()))
    delta: dict[str, tuple[object, object]] = {}
    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            delta[key] = (val_a, val_b)
    return delta


def format_comparison_report(report: RunComparisonReport) -> tuple[str, ...]:
    """Format a comparison report as human-readable lines.

    Args:
        report: Structured comparison report.

    Returns:
        Tuple of printable lines.
    """
    lines: list[str] = [
        f"Run A: {report.run_id_a}",
        f"Run B: {report.run_id_b}",
        f"State: {report.state_a} vs {report.state_b}",
        f"Dataset: {report.dataset_a} vs {report.dataset_b}",
        f"Created: {report.created_at_a} vs {report.created_at_b}",
    ]
    if report.config_delta:
        lines.append("Config differences:")
        for key, (val_a, val_b) in sorted(report.config_delta.items()):
            lines.append(f"  {key}: {val_a!r} -> {val_b!r}")
    else:
        lines.append("Config differences: none")
    return tuple(lines)
