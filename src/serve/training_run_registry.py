"""Training run lifecycle and lineage persistence.

This module stores lifecycle state transitions and lineage links under the
configured data-root so training runs remain inspectable and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.constants import (
    LINEAGE_DIR_NAME,
    LINEAGE_GRAPH_FILE_NAME,
    RUN_INDEX_FILE_NAME,
    RUN_STATE_FILE_NAME,
    RUNS_DIR_NAME,
)
from core.errors import CrucibleServeError
from serve.training_run_io import read_json_file, write_json_file
from serve.training_run_types import (
    TrainingRunEvent,
    TrainingRunRecord,
    TrainingRunState,
    run_record_from_payload,
    validate_transition,
)


class TrainingRunRegistry:
    """Persistent lifecycle and lineage registry for training runs."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root.expanduser().resolve()
        self._runs_root = self._data_root / RUNS_DIR_NAME
        self._lineage_root = self._data_root / LINEAGE_DIR_NAME
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._lineage_root.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        dataset_name: str,
        output_dir: str,
        parent_model_path: str | None,
        config_hash: str,
    ) -> TrainingRunRecord:
        """Create a new queued run record and register lineage inputs."""
        run_id = _build_run_id()
        timestamp = _utc_now_iso()
        record = TrainingRunRecord(
            run_id=run_id,
            dataset_name=dataset_name,
            output_dir=output_dir,
            parent_model_path=parent_model_path,
            config_hash=config_hash,
            state="queued",
            created_at=timestamp,
            updated_at=timestamp,
            events=(TrainingRunEvent(state="queued", timestamp=timestamp, message=None),),
        )
        self._write_run_record(record)
        self._append_index_row(run_id)
        self._append_lineage_inputs(record)
        return record

    def transition(
        self,
        run_id: str,
        next_state: TrainingRunState,
        message: str | None = None,
        artifact_contract_path: str | None = None,
        model_path: str | None = None,
    ) -> TrainingRunRecord:
        """Persist one lifecycle transition and optional artifact updates."""
        record = self._load_run_record(run_id)
        validate_transition(record.state, next_state)
        timestamp = _utc_now_iso()
        next_events = record.events + (
            TrainingRunEvent(state=next_state, timestamp=timestamp, message=message),
        )
        next_record = TrainingRunRecord(
            run_id=record.run_id,
            dataset_name=record.dataset_name,
            output_dir=record.output_dir,
            parent_model_path=record.parent_model_path,
            config_hash=record.config_hash,
            state=next_state,
            created_at=record.created_at,
            updated_at=timestamp,
            events=next_events,
            artifact_contract_path=artifact_contract_path or record.artifact_contract_path,
            error_message=message
            if next_state in {"failed", "cancelled"}
            else record.error_message,
        )
        self._write_run_record(next_record)
        if artifact_contract_path or model_path:
            self._update_lineage_outputs(
                run_id=run_id,
                artifact_contract_path=artifact_contract_path,
                model_path=model_path,
            )
        return next_record

    def load_run(self, run_id: str) -> TrainingRunRecord:
        """Load one run lifecycle record by ID."""
        return self._load_run_record(run_id)

    def list_runs(self) -> tuple[str, ...]:
        """List run IDs from lifecycle index in insertion order."""
        index_path = self._runs_root / RUN_INDEX_FILE_NAME
        payload = read_json_file(index_path, default_value={"runs": []})
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise CrucibleServeError(f"Invalid run index format at {index_path}: expected runs list.")
        return tuple(str(item) for item in payload["runs"])

    def load_lineage_graph(self) -> dict[str, object]:
        """Load current lineage graph payload."""
        runs_payload, edges_payload = self._read_lineage_graph()
        return {"runs": runs_payload, "edges": edges_payload}

    def _write_run_record(self, record: TrainingRunRecord) -> None:
        run_dir = self._runs_root / record.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["events"] = [asdict(event) for event in record.events]
        write_json_file(run_dir / RUN_STATE_FILE_NAME, payload)

    def _load_run_record(self, run_id: str) -> TrainingRunRecord:
        state_path = self._runs_root / run_id / RUN_STATE_FILE_NAME
        payload = read_json_file(state_path)
        if not isinstance(payload, dict):
            raise CrucibleServeError(f"Invalid run state payload at {state_path}: expected object.")
        return run_record_from_payload(payload, state_path)

    def _append_index_row(self, run_id: str) -> None:
        index_path = self._runs_root / RUN_INDEX_FILE_NAME
        payload = read_json_file(index_path, default_value={"runs": []})
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise CrucibleServeError(f"Invalid run index format at {index_path}: expected runs list.")
        run_ids = payload["runs"]
        if run_id not in run_ids:
            run_ids.append(run_id)
            write_json_file(index_path, payload)

    def _append_lineage_inputs(self, record: TrainingRunRecord) -> None:
        runs_payload, edges_payload = self._read_lineage_graph()
        runs_payload[record.run_id] = {
            "dataset_name": record.dataset_name,
            "output_dir": record.output_dir,
            "parent_model_path": record.parent_model_path,
            "config_hash": record.config_hash,
            "created_at": record.created_at,
            "artifact_contract_path": record.artifact_contract_path,
        }
        _append_unique_edge(
            edges_payload,
            from_node=f"dataset:{record.dataset_name}",
            to_node=f"run:{record.run_id}",
            edge_type="trained_on",
        )
        if record.parent_model_path:
            _append_unique_edge(
                edges_payload,
                from_node=f"model:{record.parent_model_path}",
                to_node=f"run:{record.run_id}",
                edge_type="initialized_from",
            )
        self._write_lineage_graph(runs_payload, edges_payload)

    def _update_lineage_outputs(
        self,
        run_id: str,
        artifact_contract_path: str | None,
        model_path: str | None,
    ) -> None:
        runs_payload, edges_payload = self._read_lineage_graph()
        run_payload = runs_payload.get(run_id)
        if isinstance(run_payload, dict):
            if artifact_contract_path:
                run_payload["artifact_contract_path"] = artifact_contract_path
            if model_path:
                run_payload["model_path"] = model_path
        if model_path:
            _append_unique_edge(
                edges_payload,
                from_node=f"run:{run_id}",
                to_node=f"model:{model_path}",
                edge_type="produced",
            )
        self._write_lineage_graph(runs_payload, edges_payload)

    def _read_lineage_graph(self) -> tuple[dict[str, dict[str, object]], list[object]]:
        graph_path = self._lineage_root / LINEAGE_GRAPH_FILE_NAME
        payload = read_json_file(graph_path, default_value={"runs": {}, "edges": []})
        if not isinstance(payload, dict):
            raise CrucibleServeError(f"Invalid lineage graph at {graph_path}: expected object.")
        runs_payload = payload.get("runs")
        edges_payload = payload.get("edges")
        if not isinstance(runs_payload, dict) or not isinstance(edges_payload, list):
            raise CrucibleServeError(f"Invalid lineage graph at {graph_path}: expected runs/edges.")
        normalized_runs: dict[str, dict[str, object]] = {}
        for run_id, run_payload in runs_payload.items():
            if isinstance(run_id, str) and isinstance(run_payload, dict):
                normalized_runs[run_id] = run_payload
        return normalized_runs, edges_payload

    def _write_lineage_graph(
        self,
        runs_payload: dict[str, dict[str, object]],
        edges_payload: list[object],
    ) -> None:
        write_json_file(
            self._lineage_root / LINEAGE_GRAPH_FILE_NAME,
            {"runs": runs_payload, "edges": edges_payload},
        )


def _build_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{timestamp}-{uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_unique_edge(
    edges_payload: list[object],
    from_node: str,
    to_node: str,
    edge_type: str,
) -> None:
    candidate = {"from": from_node, "to": to_node, "type": edge_type}
    if candidate not in edges_payload:
        edges_payload.append(candidate)
