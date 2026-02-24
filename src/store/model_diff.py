"""Model version comparison utilities.

This module provides diffing capabilities to compare metadata
and configuration between two registered model versions.
"""

from __future__ import annotations

from dataclasses import fields

from core.model_registry_types import ModelVersion
from serve.training_run_io import read_json_file


def diff_model_versions(
    version_a: ModelVersion,
    version_b: ModelVersion,
) -> dict[str, tuple[object, object]]:
    """Compare two model versions field by field.

    Returns a dictionary of fields that differ, where each value is
    a tuple of (value_from_a, value_from_b).

    Args:
        version_a: First model version to compare.
        version_b: Second model version to compare.

    Returns:
        Mapping of differing field names to value pairs.
    """
    differences: dict[str, tuple[object, object]] = {}
    for f in fields(ModelVersion):
        val_a = getattr(version_a, f.name)
        val_b = getattr(version_b, f.name)
        if val_a != val_b:
            differences[f.name] = (val_a, val_b)
    return differences


def diff_model_configs(
    version_a: ModelVersion,
    version_b: ModelVersion,
) -> dict[str, tuple[object, object]]:
    """Compare training config JSON files for two model versions.

    Loads training_config.json from each model's directory and
    compares all top-level keys. Returns only differing entries.

    Args:
        version_a: First model version with model_path set.
        version_b: Second model version with model_path set.

    Returns:
        Mapping of differing config keys to value pairs.
    """
    config_a = _load_training_config(version_a.model_path)
    config_b = _load_training_config(version_b.model_path)
    all_keys = set(config_a.keys()) | set(config_b.keys())
    differences: dict[str, tuple[object, object]] = {}
    for key in sorted(all_keys):
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            differences[key] = (val_a, val_b)
    return differences


def format_model_diff(
    diff: dict[str, tuple[object, object]],
) -> tuple[str, ...]:
    """Format a diff dictionary as human-readable lines.

    Args:
        diff: Mapping of field names to (value_a, value_b) pairs.

    Returns:
        Tuple of formatted diff lines.
    """
    if not diff:
        return ("No differences found.",)
    lines: list[str] = []
    for field_name in sorted(diff.keys()):
        val_a, val_b = diff[field_name]
        lines.append(f"  {field_name}: {val_a!r} -> {val_b!r}")
    return tuple(lines)


def _load_training_config(model_path: str) -> dict[str, object]:
    """Load training_config.json from a model directory.

    Args:
        model_path: Path to the model directory.

    Returns:
        Parsed training config dictionary, or empty dict on failure.
    """
    from pathlib import Path

    config_path = Path(model_path).parent / "training_config.json"
    try:
        raw = read_json_file(config_path, default_value={})
    except Exception:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}
