"""Ingest checkpoint persistence.

This module stores stage outputs for resumable ingest execution.
It enables resume behavior across process restarts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from core.constants import (
    CHECKPOINT_DEDUP_RECORDS_FILE_NAME,
    CHECKPOINT_ENRICHED_RECORDS_FILE_NAME,
    CHECKPOINT_SOURCE_RECORDS_FILE_NAME,
    CHECKPOINT_STATE_FILE_NAME,
    DATASETS_DIR_NAME,
    INGEST_CHECKPOINT_DIR_NAME,
)
from core.errors import CrucibleIngestError
from core.types import DataRecord, SourceTextRecord
from store.record_payload import read_data_records_jsonl, write_data_records_jsonl


@dataclass(frozen=True)
class IngestCheckpointState:
    """Checkpoint state metadata."""

    run_signature: str
    stage: str


class IngestCheckpointStore:
    """Filesystem-backed ingest checkpoint store."""

    def __init__(self, data_root: Path, dataset_name: str) -> None:
        self._checkpoint_dir = (
            data_root / DATASETS_DIR_NAME / dataset_name / INGEST_CHECKPOINT_DIR_NAME
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_run(self, run_signature: str, resume: bool) -> IngestCheckpointState:
        """Prepare checkpoint state for a new or resumed run.

        Args:
            run_signature: Deterministic run signature.
            resume: Whether this run should resume.

        Returns:
            Checkpoint state for current run.

        Raises:
            CrucibleIngestError: If resume requested without matching checkpoint.
        """
        if resume:
            state = self._read_state()
            if state is None:
                raise CrucibleIngestError(
                    "Cannot resume ingest: checkpoint state not found. "
                    "Run ingest once without --resume to initialize checkpoints."
                )
            if state.run_signature != run_signature:
                raise CrucibleIngestError(
                    "Cannot resume ingest: checkpoint does not match current options/source. "
                    "Retry without --resume or use the same ingest parameters."
                )
            return state
        self.clear()
        state = IngestCheckpointState(
            run_signature=run_signature,
            stage="initialized",
        )
        self._write_state(state)
        return state

    def clear(self) -> None:
        """Remove all checkpoint files for dataset."""
        for file_path in self._checkpoint_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()

    def has_stage(self, state: IngestCheckpointState, stage: str) -> bool:
        """Return whether checkpoint has completed the target stage."""
        return _stage_rank(state.stage) >= _stage_rank(stage)

    def update_stage(
        self,
        state: IngestCheckpointState,
        stage: str,
    ) -> IngestCheckpointState:
        """Persist an updated stage in checkpoint state."""
        updated_state = IngestCheckpointState(
            run_signature=state.run_signature,
            stage=stage,
        )
        self._write_state(updated_state)
        return updated_state

    def save_source_records(self, records: list[SourceTextRecord]) -> None:
        """Persist source records stage output."""
        _write_source_records(self._source_records_path(), records)

    def load_source_records(self) -> list[SourceTextRecord]:
        """Load source records stage output."""
        return _read_source_records(self._source_records_path())

    def save_dedup_records(self, records: list[SourceTextRecord]) -> None:
        """Persist deduplicated source records."""
        _write_source_records(self._dedup_records_path(), records)

    def load_dedup_records(self) -> list[SourceTextRecord]:
        """Load deduplicated source records."""
        return _read_source_records(self._dedup_records_path())

    def save_enriched_records(self, records: list[DataRecord]) -> None:
        """Persist enriched records stage output."""
        _write_data_records(self._enriched_records_path(), records)

    def load_enriched_records(self) -> list[DataRecord]:
        """Load enriched records stage output."""
        return _read_data_records(self._enriched_records_path())

    def _read_state(self) -> IngestCheckpointState | None:
        """Read checkpoint state file if present."""
        state_path = self._state_path()
        if not state_path.exists():
            return None
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            return IngestCheckpointState(
                run_signature=str(payload["run_signature"]),
                stage=str(payload["stage"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise CrucibleIngestError(
                f"Failed to read ingest checkpoint state at {state_path}: {error}. "
                "Delete the checkpoint directory and retry ingest."
            ) from error

    def _write_state(self, state: IngestCheckpointState) -> None:
        """Write checkpoint state file."""
        state_path = self._state_path()
        state_path.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")

    def _state_path(self) -> Path:
        return self._checkpoint_dir / CHECKPOINT_STATE_FILE_NAME

    def _source_records_path(self) -> Path:
        return self._checkpoint_dir / CHECKPOINT_SOURCE_RECORDS_FILE_NAME

    def _dedup_records_path(self) -> Path:
        return self._checkpoint_dir / CHECKPOINT_DEDUP_RECORDS_FILE_NAME

    def _enriched_records_path(self) -> Path:
        return self._checkpoint_dir / CHECKPOINT_ENRICHED_RECORDS_FILE_NAME


def _write_source_records(records_path: Path, records: list[SourceTextRecord]) -> None:
    """Write source text records to JSONL file."""
    lines = [
        json.dumps({"source_uri": record.source_uri, "text": record.text}, sort_keys=True)
        for record in records
    ]
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_source_records(records_path: Path) -> list[SourceTextRecord]:
    """Read source text records from JSONL file."""
    if not records_path.exists():
        raise CrucibleIngestError(
            f"Missing ingest checkpoint file at {records_path}. "
            "Retry without --resume to rebuild checkpoints."
        )
    records: list[SourceTextRecord] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = _parse_source_line(records_path, line, line_number)
        records.append(SourceTextRecord(source_uri=payload["source_uri"], text=payload["text"]))
    return records


def _parse_source_line(records_path: Path, line: str, line_number: int) -> dict[str, str]:
    """Parse one source checkpoint line."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise CrucibleIngestError(
            f"Failed to parse ingest checkpoint at {records_path}:{line_number}: {error.msg}. "
            "Retry without --resume to rebuild checkpoints."
        ) from error
    source_uri = payload.get("source_uri")
    text = payload.get("text")
    if not isinstance(source_uri, str) or not isinstance(text, str):
        raise CrucibleIngestError(
            f"Invalid ingest checkpoint payload at {records_path}:{line_number}. "
            "Retry without --resume to rebuild checkpoints."
        )
    return {"source_uri": source_uri, "text": text}


def _write_data_records(records_path: Path, records: list[DataRecord]) -> None:
    """Write DataRecord checkpoint payload."""
    write_data_records_jsonl(records_path, records)


def _read_data_records(records_path: Path) -> list[DataRecord]:
    """Read DataRecord checkpoint payload."""
    if not records_path.exists():
        raise CrucibleIngestError(
            f"Missing ingest checkpoint file at {records_path}. "
            "Retry without --resume to rebuild checkpoints."
        )
    try:
        return read_data_records_jsonl(records_path)
    except (OSError, ValueError) as error:
        raise CrucibleIngestError(
            f"Failed to load ingest checkpoint data at {records_path}: {error}. "
            "Retry without --resume to rebuild checkpoints."
        ) from error


def _stage_rank(stage: str) -> int:
    """Map stage name to ordering rank."""
    stage_order = {
        "initialized": 0,
        "source_loaded": 1,
        "deduplicated": 2,
        "enriched": 3,
    }
    return stage_order.get(stage, -1)
