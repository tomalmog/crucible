"""Unit tests for training lifecycle and lineage registry."""

from __future__ import annotations

import pytest

from core.errors import ForgeServeError
from serve.training_run_registry import TrainingRunRegistry


def test_start_run_persists_queued_record_and_lineage_edges(tmp_path) -> None:
    """Starting a run should persist lifecycle state and dataset lineage edge."""
    registry = TrainingRunRegistry(tmp_path)
    run_record = registry.start_run(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        parent_model_path=None,
        config_hash="abc123",
    )
    lineage = registry.load_lineage_graph()
    dataset_edge = {
        "from": "dataset:demo",
        "to": f"run:{run_record.run_id}",
        "type": "trained_on",
    }

    assert (
        run_record.state == "queued"
        and registry.list_runs() == (run_record.run_id,)
        and dataset_edge in lineage["edges"]
    )


def test_transition_rejects_invalid_state_changes(tmp_path) -> None:
    """Lifecycle transition should reject illegal state machine edges."""
    registry = TrainingRunRegistry(tmp_path)
    run_record = registry.start_run(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        parent_model_path=None,
        config_hash="abc123",
    )
    registry.transition(run_record.run_id, "running")
    registry.transition(run_record.run_id, "completed")

    with pytest.raises(ForgeServeError):
        registry.transition(run_record.run_id, "running")

    assert True


def test_transition_completed_updates_artifact_and_model_lineage(tmp_path) -> None:
    """Completed transition should capture artifact contract and model output edge."""
    registry = TrainingRunRegistry(tmp_path)
    run_record = registry.start_run(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        parent_model_path="/tmp/parent.pt",
        config_hash="abc123",
    )
    registry.transition(run_record.run_id, "running")
    completed = registry.transition(
        run_record.run_id,
        "completed",
        artifact_contract_path="/tmp/out/training_artifacts_manifest.json",
        model_path="/tmp/out/model.pt",
    )
    lineage = registry.load_lineage_graph()
    produced_edge = {
        "from": f"run:{run_record.run_id}",
        "to": "model:/tmp/out/model.pt",
        "type": "produced",
    }

    assert completed.artifact_contract_path and produced_edge in lineage["edges"]
