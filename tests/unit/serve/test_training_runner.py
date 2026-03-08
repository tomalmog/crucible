"""Unit tests for training runner dependency behavior."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from core.errors import ForgeDependencyError
from core.types import DataRecord, RecordMetadata, TrainingOptions, TrainingRunResult
from serve.training_execution import TrainingLoopResult
from serve.training_hooks import TrainingHooks
from serve.training_run_registry import TrainingRunRegistry
from serve.training_runner import run_training


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [DataRecord(record_id="id-1", text="alpha beta gamma", metadata=metadata)]


def test_run_training_raises_without_torch(monkeypatch, tmp_path) -> None:
    """Training should fail clearly when torch dependency is missing."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path))

    with pytest.raises(ForgeDependencyError):
        run_training(
            _build_records(),
            options,
            random_seed=1,
            data_root=tmp_path,
        )

    assert True


def test_run_training_persists_completed_lifecycle_record(monkeypatch, tmp_path) -> None:
    """Successful training should persist completed lifecycle metadata."""
    options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path / "out"))

    def _fake_build_context(*args, **kwargs):
        _ = args
        _ = kwargs
        return SimpleNamespace(hooks=TrainingHooks())

    def _fake_loop(context):
        _ = context
        return TrainingLoopResult(
            epoch_metrics=[],
            batch_metrics=[],
            checkpoint_dir=None,
            best_checkpoint_path=None,
            resumed_from_checkpoint=None,
        )

    def _fake_persist(*args, **kwargs):
        _ = args
        run_id = kwargs["run_id"]
        return TrainingRunResult(
            model_path=str(tmp_path / "out/model.pt"),
            history_path=str(tmp_path / "out/history.json"),
            plot_path=None,
            epochs_completed=0,
            run_id=run_id,
            artifact_contract_path=str(tmp_path / "out/training_artifacts_manifest.json"),
        )

    monkeypatch.setattr("serve.training_runner._build_runtime_context", _fake_build_context)
    monkeypatch.setattr("serve.training_runner.run_training_loop", _fake_loop)
    monkeypatch.setattr("serve.training_runner._persist_training_outputs", _fake_persist)
    result = run_training(
        records=_build_records(),
        options=options,
        random_seed=7,
        data_root=tmp_path,
    )
    run_record = TrainingRunRegistry(tmp_path).load_run(result.run_id or "")

    assert run_record.state == "completed" and result.artifact_contract_path is not None


def test_run_training_persists_failed_lifecycle_record(monkeypatch, tmp_path) -> None:
    """Training errors should transition lifecycle state to failed."""
    options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path / "out"))

    def _fake_build_context(*args, **kwargs):
        _ = args
        _ = kwargs
        return SimpleNamespace(hooks=TrainingHooks())

    def _fake_loop(context):
        _ = context
        raise RuntimeError("loop-failed")

    monkeypatch.setattr("serve.training_runner._build_runtime_context", _fake_build_context)
    monkeypatch.setattr("serve.training_runner.run_training_loop", _fake_loop)
    with pytest.raises(RuntimeError):
        run_training(
            records=_build_records(),
            options=options,
            random_seed=7,
            data_root=tmp_path,
        )
    run_id = TrainingRunRegistry(tmp_path).list_runs()[0]
    run_record = TrainingRunRegistry(tmp_path).load_run(run_id)

    assert run_record.state == "failed"
