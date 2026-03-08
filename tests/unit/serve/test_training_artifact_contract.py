"""Unit tests for training artifact contract persistence."""

from __future__ import annotations

from pathlib import Path

from core.types import TrainingRunResult
from serve.training_artifact_contract import (
    load_training_artifact_contract,
    save_training_artifact_contract,
)


def test_save_and_load_training_artifact_contract_round_trip(tmp_path: Path) -> None:
    """Artifact contract should persist and load all key training output fields."""
    result = TrainingRunResult(
        model_path=str(tmp_path / "model.pt"),
        history_path=str(tmp_path / "history.json"),
        plot_path=str(tmp_path / "plot.png"),
        epochs_completed=3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        best_checkpoint_path=str(tmp_path / "checkpoints/best.pt"),
        run_id="run-123",
    )
    model_path = Path(result.model_path)
    model_path.write_text("weights", encoding="utf-8")
    save_training_artifact_contract(
        output_dir=tmp_path,
        run_id="run-123",
        dataset_name="demo",
        parent_model_path=None,
        config_hash="abc123",
        result=result,
        tokenizer_path=str(tmp_path / "tokenizer_vocab.json"),
        training_config_path=str(tmp_path / "training_config.json"),
        reproducibility_bundle_path=str(tmp_path / "reproducibility_bundle.json"),
    )

    payload = load_training_artifact_contract(result.model_path)

    assert payload is not None and payload["run_id"] == "run-123"
