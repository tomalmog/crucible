"""Unit tests for domain adaptation runner."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from core.domain_adaptation_types import DomainAdaptationOptions
from core.errors import ForgeDependencyError
from core.types import DataRecord, RecordMetadata, TrainingRunResult
from serve.domain_adaptation_runner import run_domain_adaptation
from serve.training_execution import TrainingLoopResult
from serve.training_hooks import TrainingHooks
from serve.training_run_registry import TrainingRunRegistry


def _build_records() -> list[DataRecord]:
    """Build minimal test records."""
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [DataRecord(record_id="id-1", text="alpha beta gamma", metadata=metadata)]


def test_domain_adaptation_raises_without_torch(monkeypatch, tmp_path) -> None:
    """Domain adaptation should fail when torch is missing."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    options = DomainAdaptationOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        base_model_path="/tmp/base.pt",
    )
    with pytest.raises(ForgeDependencyError):
        run_domain_adaptation(
            _build_records(), options,
            random_seed=1, data_root=tmp_path,
            dataset_version_id="demo-v1",
        )


def test_domain_adaptation_completed_lifecycle(monkeypatch, tmp_path) -> None:
    """Successful adaptation should persist completed lifecycle record."""
    options = DomainAdaptationOptions(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        base_model_path="/tmp/base.pt",
    )

    def _fake_build_context(*args, **kwargs):
        return SimpleNamespace(hooks=TrainingHooks())

    def _fake_loop(context):
        return TrainingLoopResult(
            epoch_metrics=[], batch_metrics=[],
            checkpoint_dir=None, best_checkpoint_path=None,
            resumed_from_checkpoint=None,
        )

    def _fake_persist(context, loop_result, run_id, *args, **kwargs):
        return TrainingRunResult(
            model_path=str(tmp_path / "out/model.pt"),
            history_path=str(tmp_path / "out/history.json"),
            plot_path=None, epochs_completed=0,
            run_id=run_id,
            artifact_contract_path=str(
                tmp_path / "out/training_artifacts_manifest.json"
            ),
        )

    monkeypatch.setattr(
        "serve.domain_adaptation_runner._build_adaptation_context",
        _fake_build_context,
    )
    monkeypatch.setattr(
        "serve.domain_adaptation_runner.run_training_loop",
        _fake_loop,
    )
    monkeypatch.setattr(
        "serve.domain_adaptation_runner._persist_adaptation_outputs",
        _fake_persist,
    )
    result = run_domain_adaptation(
        records=_build_records(), options=options,
        random_seed=7, data_root=tmp_path,
        dataset_version_id="demo-v1",
    )
    run_record = TrainingRunRegistry(tmp_path).load_run(result.run_id or "")
    assert run_record.state == "completed"
    assert result.artifact_contract_path is not None


def test_domain_adaptation_failed_lifecycle(monkeypatch, tmp_path) -> None:
    """Adaptation errors should transition lifecycle state to failed."""
    options = DomainAdaptationOptions(
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        base_model_path="/tmp/base.pt",
    )

    def _fake_build_context(*args, **kwargs):
        return SimpleNamespace(hooks=TrainingHooks())

    def _fake_loop(context):
        raise RuntimeError("loop-failed")

    monkeypatch.setattr(
        "serve.domain_adaptation_runner._build_adaptation_context",
        _fake_build_context,
    )
    monkeypatch.setattr(
        "serve.domain_adaptation_runner.run_training_loop",
        _fake_loop,
    )
    with pytest.raises(RuntimeError):
        run_domain_adaptation(
            records=_build_records(), options=options,
            random_seed=7, data_root=tmp_path,
            dataset_version_id="demo-v1",
        )
    run_id = TrainingRunRegistry(tmp_path).list_runs()[0]
    run_record = TrainingRunRegistry(tmp_path).load_run(run_id)
    assert run_record.state == "failed"
