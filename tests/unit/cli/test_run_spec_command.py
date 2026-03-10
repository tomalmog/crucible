"""Unit tests for run-spec CLI execution."""

from __future__ import annotations

import pytest

from cli.main import main
from core.errors import CrucibleRunSpecError
from core.types import IngestOptions, TrainingRunResult
from store.dataset_sdk import CrucibleClient
from tests.fixture_paths import fixture_path


def test_cli_run_spec_executes_ingest_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run-spec command should route ingest step to SDK operation."""
    captured: dict[str, object] = {}

    def _fake_ingest(self: CrucibleClient, options: IngestOptions) -> str:
        captured["ingest_dataset"] = options.dataset_name
        captured["ingest_source"] = options.source_uri
        return options.dataset_name

    monkeypatch.setattr(CrucibleClient, "ingest", _fake_ingest)
    exit_code = main(["run-spec", str(fixture_path("run_spec/valid_pipeline.yaml"))])
    output = capsys.readouterr().out.strip().splitlines()

    assert (
        exit_code == 0
        and len(output) >= 1
        and captured.get("ingest_dataset") == "demo"
    )


def test_cli_run_spec_missing_dataset_raises_error() -> None:
    """Run-spec should fail when a dataset-dependent step has no dataset."""
    with pytest.raises(CrucibleRunSpecError):
        main(["run-spec", str(fixture_path("run_spec/missing_dataset.yaml"))])
    assert True


def test_cli_run_spec_train_step_parses_optimizer_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run-spec train step should forward optimizer and scheduler options."""
    captured: dict[str, object] = {}

    def _fake_train(self: CrucibleClient, options) -> TrainingRunResult:
        captured["precision_mode"] = options.precision_mode
        captured["optimizer_type"] = options.optimizer_type
        captured["weight_decay"] = options.weight_decay
        captured["scheduler_type"] = options.scheduler_type
        captured["scheduler_t_max_epochs"] = options.scheduler_t_max_epochs
        captured["progress_log_interval_steps"] = options.progress_log_interval_steps
        return TrainingRunResult(
            model_path="/tmp/model.pt",
            history_path="/tmp/history.json",
            plot_path=None,
            epochs_completed=2,
        )

    monkeypatch.setattr(CrucibleClient, "train", _fake_train)
    exit_code = main(["run-spec", str(fixture_path("run_spec/train_pipeline.yaml"))])
    output = capsys.readouterr().out.strip().splitlines()

    assert (
        exit_code == 0
        and output[0].startswith("model_path=")
        and captured
        == {
            "precision_mode": "bf16",
            "optimizer_type": "adamw",
            "weight_decay": 0.01,
            "scheduler_type": "cosine",
            "scheduler_t_max_epochs": 8,
            "progress_log_interval_steps": 4,
        }
    )


def test_cli_run_spec_supports_hardware_profile_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run-spec should execute hardware-profile step without dataset defaults."""

    def _fake_profile(self: CrucibleClient) -> dict[str, object]:
        return {"accelerator": "cpu", "gpu_count": 0}

    monkeypatch.setattr(CrucibleClient, "hardware_profile", _fake_profile)
    exit_code = main(["run-spec", str(fixture_path("run_spec/hardware_profile.yaml"))])
    output = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0 and output == ["accelerator=cpu", "gpu_count=0"]
