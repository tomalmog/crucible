"""Reproducibility bundle persistence for training runs.

This module stores environment and configuration details needed to replay a
training run deterministically in a future environment.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from core.constants import DEFAULT_REPRODUCIBILITY_BUNDLE_FILE_NAME
from core.errors import CrucibleServeError


def save_reproducibility_bundle(
    output_dir: Path,
    run_id: str,
    dataset_name: str,
    config_hash: str,
    random_seed: int,
    training_options: Mapping[str, object],
) -> Path:
    """Persist one reproducibility bundle beside training artifacts."""
    bundle_path = output_dir / DEFAULT_REPRODUCIBILITY_BUNDLE_FILE_NAME
    payload = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "config_hash": config_hash,
        "random_seed": random_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "training_options": dict(training_options),
    }
    try:
        bundle_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise CrucibleServeError(
            f"Failed to write reproducibility bundle at {bundle_path}: {error}. "
            "Check output directory permissions and retry."
        ) from error
    return bundle_path
