"""Typed training lifecycle models and validation helpers.

This module defines lifecycle states and payload parsing used by run registry
and SDK calls that inspect persisted training execution metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from core.errors import CrucibleServeError

TrainingRunState = Literal[
    "queued",
    "running",
    "checkpointing",
    "completed",
    "failed",
    "cancelled",
]
ALLOWED_STATE_TRANSITIONS: dict[TrainingRunState, tuple[TrainingRunState, ...]] = {
    "queued": ("running", "failed", "cancelled"),
    "running": ("checkpointing", "completed", "failed", "cancelled"),
    "checkpointing": ("running", "completed", "failed", "cancelled"),
    "completed": (),
    "failed": (),
    "cancelled": (),
}


@dataclass(frozen=True)
class TrainingRunEvent:
    """One lifecycle state transition event."""

    state: TrainingRunState
    timestamp: str
    message: str | None


@dataclass(frozen=True)
class TrainingRunRecord:
    """Persisted training lifecycle metadata."""

    run_id: str
    dataset_name: str
    output_dir: str
    parent_model_path: str | None
    config_hash: str
    state: TrainingRunState
    created_at: str
    updated_at: str
    events: tuple[TrainingRunEvent, ...]
    artifact_contract_path: str | None = None
    error_message: str | None = None


def validate_transition(current: TrainingRunState, next_state: TrainingRunState) -> None:
    """Validate one lifecycle transition against allowed state machine edges."""
    allowed_states = ALLOWED_STATE_TRANSITIONS[current]
    if next_state not in allowed_states:
        raise CrucibleServeError(
            f"Invalid training run state transition {current!r} -> {next_state!r}. "
            f"Allowed: {', '.join(allowed_states) or 'none'}."
        )


def run_record_from_payload(payload: dict[str, object], payload_path: Path) -> TrainingRunRecord:
    """Deserialize a lifecycle record payload from JSON."""
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise CrucibleServeError(f"Invalid run state at {payload_path}: events must be a list.")
    events = tuple(run_event_from_payload(item, payload_path) for item in raw_events)
    state = parse_state(payload.get("state"), payload_path)
    try:
        return TrainingRunRecord(
            run_id=str(payload["run_id"]),
            dataset_name=str(payload["dataset_name"]),
            output_dir=str(payload["output_dir"]),
            parent_model_path=optional_string(payload.get("parent_model_path")),
            config_hash=str(payload["config_hash"]),
            state=state,
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            events=events,
            artifact_contract_path=optional_string(payload.get("artifact_contract_path")),
            error_message=optional_string(payload.get("error_message")),
        )
    except KeyError as error:
        raise CrucibleServeError(
            f"Invalid run state at {payload_path}: missing required field {error.args[0]!r}."
        ) from error


def run_event_from_payload(payload: object, payload_path: Path) -> TrainingRunEvent:
    """Deserialize one run event payload from JSON."""
    if not isinstance(payload, dict):
        raise CrucibleServeError(f"Invalid run event at {payload_path}: expected object entries.")
    return TrainingRunEvent(
        state=parse_state(payload.get("state"), payload_path),
        timestamp=str(payload.get("timestamp", "")),
        message=optional_string(payload.get("message")),
    )


def parse_state(raw_state: object, payload_path: Path) -> TrainingRunState:
    """Parse one training state value from persisted payload."""
    if isinstance(raw_state, str) and raw_state in ALLOWED_STATE_TRANSITIONS:
        return cast(TrainingRunState, raw_state)
    allowed = ", ".join(ALLOWED_STATE_TRANSITIONS.keys())
    raise CrucibleServeError(f"Invalid run state at {payload_path}: expected one of {allowed}.")


def optional_string(raw_value: object) -> str | None:
    """Convert optional payload field to string when present."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return raw_value
    return str(raw_value)
