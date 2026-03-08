"""Training artifact contract persistence.

This module writes a stable manifest for every training run so downstream
systems can consume artifacts without guessing file names or paths.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.constants import (
    DEFAULT_TOKENIZER_VOCAB_FILE_NAME,
    DEFAULT_TRAINING_CONFIG_FILE_NAME,
    TRAINING_ARTIFACT_CONTRACT_FILE_NAME,
)
from core.errors import ForgeServeError
from core.types import TrainingRunResult


@dataclass(frozen=True)
class TrainingArtifactContract:
    """Stable manifest of all artifacts produced by one training run."""

    run_id: str
    dataset_name: str
    parent_model_path: str | None
    config_hash: str
    created_at: str
    model_path: str
    history_path: str
    plot_path: str | None
    tokenizer_path: str
    training_config_path: str
    checkpoint_dir: str | None
    best_checkpoint_path: str | None
    reproducibility_bundle_path: str | None
    logs_path: str | None
    benchmark_results_path: str | None


def save_training_artifact_contract(
    output_dir: Path,
    run_id: str,
    dataset_name: str,
    parent_model_path: str | None,
    config_hash: str,
    result: TrainingRunResult,
    tokenizer_path: str | None = None,
    training_config_path: str | None = None,
    reproducibility_bundle_path: str | None = None,
) -> Path:
    """Persist training artifact contract to output directory."""
    manifest_path = output_dir / TRAINING_ARTIFACT_CONTRACT_FILE_NAME
    payload = TrainingArtifactContract(
        run_id=run_id,
        dataset_name=dataset_name,
        parent_model_path=parent_model_path,
        config_hash=config_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_path=result.model_path,
        history_path=result.history_path,
        plot_path=result.plot_path,
        tokenizer_path=tokenizer_path or str(output_dir / DEFAULT_TOKENIZER_VOCAB_FILE_NAME),
        training_config_path=(
            training_config_path or str(output_dir / DEFAULT_TRAINING_CONFIG_FILE_NAME)
        ),
        checkpoint_dir=result.checkpoint_dir,
        best_checkpoint_path=result.best_checkpoint_path,
        reproducibility_bundle_path=reproducibility_bundle_path,
        logs_path=None,
        benchmark_results_path=None,
    )
    _write_payload(manifest_path, asdict(payload))
    return manifest_path


def load_training_artifact_contract(model_path: str) -> dict[str, object] | None:
    """Load training artifact contract from model output directory."""
    output_dir = Path(model_path).expanduser().resolve().parent
    contract_path = output_dir / TRAINING_ARTIFACT_CONTRACT_FILE_NAME
    if not contract_path.exists():
        return None
    payload = _read_payload(contract_path)
    if not isinstance(payload, dict):
        raise ForgeServeError(
            f"Invalid artifact contract at {contract_path}: expected JSON object."
        )
    return payload


def _write_payload(payload_path: Path, payload: dict[str, object]) -> None:
    try:
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise ForgeServeError(
            f"Failed to write artifact contract at {payload_path}: {error}. "
            "Check directory permissions and retry."
        ) from error


def _read_payload(payload_path: Path) -> object:
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ForgeServeError(
            f"Failed to parse artifact contract at {payload_path}: {error.msg}."
        ) from error
    except OSError as error:
        raise ForgeServeError(
            f"Failed to read artifact contract at {payload_path}: {error}."
        ) from error
