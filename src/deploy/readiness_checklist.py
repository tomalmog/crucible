"""Deployment readiness checklist.

This module validates that a model and its supporting artifacts
are ready for deployment by running a series of gate checks.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.deployment_types import DeploymentChecklist
from core.errors import ForgeDeployError

_MAX_REASONABLE_MODEL_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def run_readiness_checklist(
    model_path: str, output_dir: str,
) -> DeploymentChecklist:
    """Run all deployment readiness gate checks.

    Args:
        model_path: Path to the model file.
        output_dir: Directory where deployment artifacts reside.

    Returns:
        DeploymentChecklist with per-check pass/fail status.
    """
    items: list[tuple[str, bool]] = []

    items.append(("model_exists", os.path.isfile(model_path)))
    items.append(("output_dir_exists", os.path.isdir(output_dir)))

    config_exists = _check_config_exists(output_dir)
    items.append(("config_exists", config_exists))

    tokenizer_exists = _check_tokenizer_exists(output_dir)
    items.append(("tokenizer_exists", tokenizer_exists))

    model_loadable = _check_model_loadable(model_path)
    items.append(("model_loadable", model_loadable))

    size_ok = _check_file_size_reasonable(model_path)
    items.append(("file_size_reasonable", size_ok))

    items_tuple = tuple(items)
    all_passed = all(passed for _, passed in items_tuple)

    return DeploymentChecklist(
        items=items_tuple, all_passed=all_passed,
    )


def _check_config_exists(output_dir: str) -> bool:
    """Check whether a config file exists in the output directory.

    Args:
        output_dir: Directory to search.

    Returns:
        True if any common config file is found.
    """
    config_names = ("config.json", "config.yaml", "config.yml")
    return any(
        os.path.isfile(os.path.join(output_dir, name))
        for name in config_names
    )


def _check_tokenizer_exists(output_dir: str) -> bool:
    """Check whether a tokenizer file exists in the output directory.

    Args:
        output_dir: Directory to search.

    Returns:
        True if any common tokenizer file is found.
    """
    tokenizer_names = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
    )
    return any(
        os.path.isfile(os.path.join(output_dir, name))
        for name in tokenizer_names
    )


def _check_model_loadable(model_path: str) -> bool:
    """Check whether the model file can be opened and read.

    Args:
        model_path: Path to the model file.

    Returns:
        True if the file can be opened and is non-empty.
    """
    try:
        with open(model_path, "rb") as fh:
            header = fh.read(16)
            return len(header) > 0
    except (OSError, IOError):
        return False


def _check_file_size_reasonable(model_path: str) -> bool:
    """Check whether the model file size is within a reasonable range.

    Args:
        model_path: Path to the model file.

    Returns:
        True if the file exists and is under the size limit.
    """
    try:
        size = os.path.getsize(model_path)
        return 0 < size <= _MAX_REASONABLE_MODEL_SIZE_BYTES
    except OSError:
        return False


def format_checklist(
    checklist: DeploymentChecklist,
) -> tuple[str, ...]:
    """Format a checklist as human-readable pass/fail lines.

    Args:
        checklist: The deployment checklist to format.

    Returns:
        Tuple of formatted status lines.
    """
    lines: list[str] = ["Deployment Readiness Checklist", "=" * 35]

    for name, passed in checklist.items:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {name}")

    verdict = "ALL CHECKS PASSED" if checklist.all_passed else "SOME CHECKS FAILED"
    lines.append("")
    lines.append(f"Result: {verdict}")

    return tuple(lines)
