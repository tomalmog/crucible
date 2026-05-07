"""Runner for curated model health suites."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import cast

from core.activation_patching_types import ActivationPatchingOptions
from core.activation_pca_types import ActivationPcaOptions
from core.linear_probe_types import LinearProbeOptions
from core.logit_lens_types import LogitLensOptions
from serve.model_health_report import (
    HealthCheckRunReport,
    build_suite_report,
    completed_check_report,
    failed_check_report,
)
from serve.model_health_suite import (
    ModelHealthCheckCommand,
    ModelHealthCheckOptions,
    build_model_health_check_commands,
)


def run_model_health_suite(
    suite_id: str,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    """Run all checks in a suite and return one JSON-ready report."""
    commands = build_model_health_check_commands(suite_id, options)
    check_reports = tuple(_run_check(command, options, records) for command in commands)
    return build_suite_report(suite_id, options, check_reports)


def load_health_records_from_data_root(data_root: Path, dataset_name: str) -> list[object]:
    """Load calibration records for a model health suite."""
    if not dataset_name.strip():
        return []
    from dataclasses import replace

    from core.config import CrucibleConfig
    from store.dataset_sdk import CrucibleClient

    config = replace(CrucibleConfig.from_env(), data_root=data_root)
    _, records = CrucibleClient(config).dataset(dataset_name).load_records()
    return list(records)


def _run_check(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> HealthCheckRunReport:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_single_check(command, options, records)
        return completed_check_report(command, result)
    except Exception as exc:
        return failed_check_report(command, exc)


def _run_single_check(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    if command.tool_name == "weight-norms":
        return _run_weight_norms(command, options)
    if command.tool_name == "activation-norms":
        return _run_activation_norms(command, options, records)
    if command.tool_name == "gradient-norms":
        return _run_gradient_norms(command, options, records)
    if command.tool_name == "logit-lens":
        return _run_logit_lens(command, options)
    if command.tool_name == "activation-pca":
        return _run_activation_pca(command, options, records)
    if command.tool_name == "activation-patching":
        return _run_activation_patching(command, options)
    if command.tool_name in ("linear-probe", "linear-probe-layers"):
        return _run_linear_probe(command, options, records)
    raise ValueError(f"Unknown model health check: {command.tool_name}")


def _run_weight_norms(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
) -> dict[str, object]:
    from serve.model_health_diagnostics import run_weight_norm_scan

    return run_weight_norm_scan(
        options.model_path,
        _base_model(options),
        _str_arg(command.args, "layer_indices"),
    )


def _run_activation_norms(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    from serve.model_health_diagnostics import run_activation_norm_scan

    return run_activation_norm_scan(
        options.model_path,
        _base_model(options),
        records,
        options.max_samples,
        _str_arg(command.args, "layer_indices"),
    )


def _run_gradient_norms(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    from serve.model_health_diagnostics import run_gradient_norm_scan

    return run_gradient_norm_scan(
        options.model_path,
        _base_model(options),
        records,
        options.max_samples,
        _str_arg(command.args, "layer_indices"),
    )


def _run_logit_lens(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
) -> dict[str, object]:
    from serve.logit_lens_runner import run_logit_lens

    lens_options = LogitLensOptions(
        model_path=options.model_path,
        output_dir=_str_arg(command.args, "output_dir"),
        input_text=options.probe_text,
        base_model=_base_model(options),
        top_k=5,
        layer_indices=_str_arg(command.args, "layer_indices"),
    )
    return cast(dict[str, object], run_logit_lens(lens_options))


def _run_activation_pca(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    from serve.activation_pca_runner import run_activation_pca

    pca_options = ActivationPcaOptions(
        model_path=options.model_path,
        output_dir=_str_arg(command.args, "output_dir"),
        dataset_name=options.dataset_name,
        base_model=_base_model(options),
        layer_index=_first_layer_index(_str_arg(command.args, "layer_indices")),
        max_samples=options.max_samples,
        granularity="sample",
        color_field="",
    )
    return cast(dict[str, object], run_activation_pca(pca_options, records))


def _run_activation_patching(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
) -> dict[str, object]:
    from serve.activation_patching_runner import run_activation_patching

    patch_options = ActivationPatchingOptions(
        model_path=options.model_path,
        output_dir=_str_arg(command.args, "output_dir"),
        clean_text=options.clean_text,
        corrupted_text=options.corrupted_text,
        target_token_index=-1,
        base_model=_base_model(options),
        metric="logit_diff",
    )
    return cast(dict[str, object], run_activation_patching(patch_options))


def _run_linear_probe(
    command: ModelHealthCheckCommand,
    options: ModelHealthCheckOptions,
    records: list[object],
) -> dict[str, object]:
    from serve.linear_probe_runner import run_linear_probe

    probe_options = LinearProbeOptions(
        model_path=options.model_path,
        output_dir=_str_arg(command.args, "output_dir"),
        dataset_name=options.dataset_name,
        label_field=options.label_field,
        base_model=_base_model(options),
        layer_index=_probe_layer_index(command, options),
        max_samples=options.max_samples,
        epochs=8,
        learning_rate=0.001,
    )
    return cast(dict[str, object], run_linear_probe(probe_options, records))


def _base_model(options: ModelHealthCheckOptions) -> str | None:
    return options.base_model.strip() or None


def _str_arg(args: dict[str, object], key: str) -> str:
    value = args.get(key)
    return value if isinstance(value, str) else ""


def _first_layer_index(layer_indices: str) -> int:
    if not layer_indices.strip():
        return -1
    first = layer_indices.split(",", 1)[0].strip()
    if "-" in first:
        first = first.split("-", 1)[0].strip()
    return int(first)


def _probe_layer_index(command: ModelHealthCheckCommand, options: ModelHealthCheckOptions) -> int:
    if command.tool_name == "linear-probe-layers" and not options.layer_indices.strip():
        return -2
    return _first_layer_index(options.layer_indices)
