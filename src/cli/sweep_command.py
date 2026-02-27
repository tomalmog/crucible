"""Sweep command wiring for Forge CLI.

This module isolates hyperparameter sweep command parser and execution
logic, mapping CLI arguments to SweepConfig for sweep orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeSweepError
from core.sweep_types import SweepConfig, SweepParameter
from serve.sweep_analysis import format_sweep_report, format_sweep_report_json
from serve.sweep_runner import run_sweep
from store.dataset_sdk import ForgeClient


def add_sweep_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register sweep subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "sweep",
        help="Run hyperparameter sweep over training configurations",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for sweep trials",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="YAML file with sweep parameter definitions",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Inline JSON parameter definitions (alternative to --config-file)",
    )
    parser.add_argument(
        "--strategy",
        default="grid",
        choices=["grid", "random"],
        help="Sweep strategy (grid or random)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=10,
        help="Maximum number of trials for random search",
    )
    parser.add_argument(
        "--metric",
        default="validation_loss",
        help="Metric name to optimize",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=False,
        help="Maximize metric instead of minimizing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )


def run_sweep_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle sweep command invocation.

    Args:
        client: SDK client for training.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    import json as _json
    config = _build_sweep_config(args)
    result = run_sweep(client, config, random_seed=42)
    if getattr(args, "json", False):
        print(_json.dumps(format_sweep_report_json(result)))
    else:
        report = format_sweep_report(result)
        print(report)
    return 0


def _build_sweep_config(args: argparse.Namespace) -> SweepConfig:
    """Build SweepConfig from CLI arguments and config source.

    Supports YAML file (--config-file) or inline JSON (--params).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Validated SweepConfig.

    Raises:
        ForgeSweepError: If config is invalid or missing.
    """
    if args.params:
        parameters = _parse_inline_params(args.params)
    elif args.config_file:
        parameters = _load_parameters_from_yaml(args.config_file)
    else:
        raise ForgeSweepError(
            "Sweep requires --config-file or --params. "
            "Provide parameter definitions via one of these flags."
        )
    return SweepConfig(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        base_output_dir=args.output_dir,
        parameters=parameters,
        strategy=args.strategy,
        max_trials=args.max_trials,
        metric=args.metric,
        minimize=not args.maximize,
    )


def _parse_inline_params(params_json: str) -> tuple[SweepParameter, ...]:
    """Parse sweep parameters from inline JSON string.

    Expected format: {"parameters": [{"name": "learning_rate", "values": [0.001, 0.01]}]}

    Args:
        params_json: JSON string with parameter definitions.

    Returns:
        Tuple of SweepParameter instances.
    """
    import json as _json
    try:
        raw = _json.loads(params_json)
    except _json.JSONDecodeError as error:
        raise ForgeSweepError(
            f"Failed to parse --params JSON: {error}."
        ) from error
    return _parse_parameters(raw)


def _load_parameters_from_yaml(config_path: str) -> tuple[SweepParameter, ...]:
    """Load sweep parameters from a YAML config file.

    Args:
        config_path: Path to YAML sweep config.

    Returns:
        Tuple of SweepParameter instances.

    Raises:
        ForgeDependencyError: If PyYAML is unavailable.
        ForgeSweepError: If file is invalid.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise ForgeDependencyError(
            "Sweep config requires PyYAML. Install with 'pip install pyyaml==6.0.2'."
        ) from error
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ForgeSweepError(
            f"Sweep config file not found at {path}."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ForgeSweepError(
            f"Failed to parse sweep config: {error}."
        ) from error
    return _parse_parameters(raw)


def _parse_parameters(raw: Any) -> tuple[SweepParameter, ...]:
    """Parse SweepParameter instances from raw YAML data.

    Args:
        raw: Parsed YAML dictionary.

    Returns:
        Tuple of SweepParameter instances.

    Raises:
        ForgeSweepError: If data structure is invalid.
    """
    if not isinstance(raw, dict):
        raise ForgeSweepError("Sweep config must be a YAML mapping.")
    params_list = raw.get("parameters", [])
    if not isinstance(params_list, list) or not params_list:
        raise ForgeSweepError(
            "Sweep config must contain a non-empty 'parameters' list."
        )
    results: list[SweepParameter] = []
    for entry in params_list:
        results.append(_parse_single_parameter(entry))
    return tuple(results)


def _parse_single_parameter(entry: Any) -> SweepParameter:
    """Parse one SweepParameter from a YAML entry.

    Args:
        entry: Dictionary from YAML parameters list.

    Returns:
        SweepParameter instance.

    Raises:
        ForgeSweepError: If entry is malformed.
    """
    if not isinstance(entry, dict) or "name" not in entry:
        raise ForgeSweepError(
            "Each sweep parameter must be a mapping with a 'name' field."
        )
    values = tuple(float(v) for v in entry.get("values", []))
    return SweepParameter(
        name=entry["name"],
        values=values,
        min_value=float(entry.get("min_value", 0.0)),
        max_value=float(entry.get("max_value", 1.0)),
        log_scale=bool(entry.get("log_scale", False)),
    )
