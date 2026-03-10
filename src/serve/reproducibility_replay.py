"""Reproducibility replay for training runs.

This module loads a previously saved reproducibility bundle and
reconstructs TrainingOptions so the training run can be re-executed
with identical configuration.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.errors import CrucibleServeError
from core.types import TrainingOptions, TrainingRunResult


def load_reproducibility_bundle(bundle_path: str) -> dict[str, object]:
    """Read and validate a reproducibility bundle JSON file.

    Args:
        bundle_path: Absolute or relative path to the bundle file.

    Returns:
        Parsed bundle dictionary.

    Raises:
        CrucibleServeError: If the file is missing, unreadable, or invalid.
    """
    path = Path(bundle_path)
    if not path.exists():
        raise CrucibleServeError(
            f"Reproducibility bundle not found at {bundle_path}. "
            "Verify the path and retry."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CrucibleServeError(
            f"Failed to read reproducibility bundle at {bundle_path}: {error}."
        ) from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise CrucibleServeError(
            f"Invalid JSON in reproducibility bundle at {bundle_path}: {error}."
        ) from error
    if not isinstance(data, dict):
        raise CrucibleServeError(
            f"Reproducibility bundle at {bundle_path} is not a JSON object."
        )
    if "training_options" not in data:
        raise CrucibleServeError(
            f"Reproducibility bundle at {bundle_path} is missing "
            "'training_options' key."
        )
    return data


def reconstruct_training_options(
    bundle: dict[str, object],
) -> TrainingOptions:
    """Build TrainingOptions from a reproducibility bundle.

    Args:
        bundle: Parsed bundle dictionary with a 'training_options' key.

    Returns:
        Reconstructed TrainingOptions instance.

    Raises:
        CrucibleServeError: If the training_options dict is incompatible.
    """
    raw_options = bundle.get("training_options")
    if not isinstance(raw_options, dict):
        raise CrucibleServeError(
            "Bundle 'training_options' is not a dictionary. "
            "Cannot reconstruct TrainingOptions."
        )
    try:
        return TrainingOptions(**raw_options)
    except TypeError as error:
        raise CrucibleServeError(
            f"Cannot reconstruct TrainingOptions from bundle: {error}. "
            "The bundle may be from an incompatible version."
        ) from error


def replay_training_run(
    client: Any,
    bundle_path: str,
    output_dir: str | None = None,
) -> TrainingRunResult:
    """Replay a training run from a reproducibility bundle.

    Loads the bundle, reconstructs TrainingOptions, optionally overrides
    the output directory, and delegates to ``client.train()``.

    Args:
        client: CrucibleClient instance used to execute training.
        bundle_path: Path to the reproducibility bundle JSON file.
        output_dir: Optional override for the training output directory.

    Returns:
        TrainingRunResult from the replayed training run.

    Raises:
        CrucibleServeError: If bundle loading or option reconstruction fails.
    """
    bundle = load_reproducibility_bundle(bundle_path)
    options = reconstruct_training_options(bundle)
    if output_dir is not None:
        options = replace(options, output_dir=output_dir)
    return client.train(options)
