"""Tests for curated model health suite command construction."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

import serve.model_health_runner as model_health_runner
from serve.model_health_runner import run_model_health_suite
from serve.model_health_suite import (
    ModelHealthCheckCommand,
    ModelHealthCheckOptions,
    build_model_health_check_commands,
    validate_model_health_options,
)


def _options(label_field: str = "label") -> ModelHealthCheckOptions:
    return ModelHealthCheckOptions(
        model_path="/models/candidate.pt",
        dataset_name="calibration",
        probe_text="The customer asked for a refund because",
        clean_text="The item arrived broken.",
        corrupted_text="The item worked as expected.",
        label_field=label_field,
    )


def test_standard_suite_builds_four_checks() -> None:
    commands = build_model_health_check_commands("standard", _options())
    assert len(commands) == 4


def test_supervised_suite_adds_layer_wise_linear_probe() -> None:
    commands = build_model_health_check_commands("supervised", _options())
    assert any(command.tool_name == "linear-probe-layers" for command in commands)


def test_supervised_suite_requires_label_field() -> None:
    missing = validate_model_health_options("supervised", _options(label_field=""))
    assert missing == ("label_field",)


def test_activation_patching_uses_remote_job_alias() -> None:
    commands = build_model_health_check_commands("standard", _options())
    patching = next(command for command in commands if command.tool_name == "activation-patching")
    assert patching.remote_job_type == "activation-patch"


def test_remote_args_include_model_path() -> None:
    command = build_model_health_check_commands("standard", _options())[0]
    payload = json.loads(command.remote_args_json("/models/candidate.pt"))
    assert payload["model_path"] == "/models/candidate.pt"


def test_health_command_config_marks_report_workflow() -> None:
    command = build_model_health_check_commands("standard", _options())[0]
    assert command.config["workflow"] == "model-health-check"


def test_model_health_report_contains_all_standard_checks(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_health_runner, "_run_single_check", _fake_run_single_check)
    report = run_model_health_suite("standard", _options_with_output(tmp_path), [])
    assert len(report["checks"]) == 4


def test_model_health_report_includes_plain_english_summary(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_health_runner, "_run_single_check", _fake_run_single_check)
    report = run_model_health_suite("standard", _options_with_output(tmp_path), [])
    assert "Completed 4 of 4 model-health checks" in str(report["plain_english_summary"])


def test_model_health_report_includes_recommended_actions(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_health_runner, "_run_single_check", _fake_run_single_check)
    report = run_model_health_suite("standard", _options_with_output(tmp_path), [])
    first_check = report["checks"][0]
    assert "recommended_action" in first_check


def test_targeted_suite_uses_selected_checks() -> None:
    options = ModelHealthCheckOptions(
        model_path="/models/candidate.pt",
        dataset_name="",
        probe_text="",
        clean_text="",
        corrupted_text="",
        check_ids=("weight-norms",),
    )
    commands = build_model_health_check_commands("targeted", options)
    assert commands[0].tool_name == "weight-norms"


def test_model_health_report_writes_single_report_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_health_runner, "_run_single_check", _fake_run_single_check)
    run_model_health_suite("standard", _options_with_output(tmp_path), [])
    assert (tmp_path / "model_health_report.json").exists()


def test_model_health_report_marks_failed_checks_for_review(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_health_runner, "_run_single_check", _failing_run_single_check)
    report = run_model_health_suite("standard", _options_with_output(tmp_path), [])
    assert report["status"] == "completed_with_errors"


def _options_with_output(output_dir: Path) -> ModelHealthCheckOptions:
    options = _options()
    return ModelHealthCheckOptions(
        model_path=options.model_path,
        dataset_name=options.dataset_name,
        probe_text=options.probe_text,
        clean_text=options.clean_text,
        corrupted_text=options.corrupted_text,
        label_field=options.label_field,
        output_dir=str(output_dir),
    )


def _fake_run_single_check(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    if command.tool_name == "weight-norms":
        return {"flagged_layer_count": 0, "max_layer_norm_ratio": 1.2}
    if command.tool_name == "activation-norms":
        return {"flagged_layer_count": 0}
    if command.tool_name == "gradient-norms":
        return {"flagged_layer_count": 0}
    if command.tool_name == "logit-lens":
        return {"input_tokens": ["The"], "layers": [{"layer_index": 0}]}
    if command.tool_name == "activation-pca":
        return {"points": [{"x": 0.1}], "explained_variance": [0.6, 0.2]}
    if command.tool_name == "activation-patching":
        return {"layer_results": [{"layer_index": 2, "recovery": 0.75}]}
    return {"layers": [{"layer_index": 1, "accuracy": 0.8}]}


def _failing_run_single_check(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    if command.tool_name == "activation-pca":
        raise RuntimeError("dataset did not produce activations")
    return _fake_run_single_check(command, options, records)
