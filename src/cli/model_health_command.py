"""CLI command for curated model health reports."""

from __future__ import annotations

import argparse
import json

from serve.model_health_runner import run_model_health_suite
from serve.model_health_suite import ModelHealthCheckOptions, validate_model_health_options
from store.dataset_sdk import CrucibleClient


def add_model_health_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the model-health-check subcommand."""
    parser = subparsers.add_parser(
        "model-health-check",
        help="Run a fixed model health suite and emit one report",
    )
    parser.add_argument("--model-path", required=True, help="Model path or HF model ID")
    parser.add_argument("--dataset", default="", help="Calibration dataset name")
    parser.add_argument("--suite", default="standard", choices=["standard", "deep", "supervised", "targeted"])
    parser.add_argument("--probe-text", default="", help="Text for prediction trace")
    parser.add_argument("--clean-text", default="", help="Expected behavior contrast")
    parser.add_argument("--corrupted-text", default="", help="Corrupted behavior contrast")
    parser.add_argument("--label-field", default="", help="Dataset label field for supervised suite")
    parser.add_argument("--max-samples", type=int, default=300, help="Max calibration samples")
    parser.add_argument("--base-model", default="", help="Base model for adapter models")
    parser.add_argument("--checks", default="", help="Comma-separated targeted check IDs")
    parser.add_argument("--layer-indices", default="", help="Layer indices, e.g. 0,3,7 or 4-8")
    parser.add_argument("--output-dir", default="./outputs/model-health")


def run_model_health_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle model-health-check command invocation."""
    options = ModelHealthCheckOptions(
        model_path=args.model_path,
        dataset_name=args.dataset,
        probe_text=args.probe_text,
        clean_text=args.clean_text,
        corrupted_text=args.corrupted_text,
        label_field=args.label_field,
        max_samples=args.max_samples,
        base_model=args.base_model,
        output_dir=args.output_dir,
        check_ids=_parse_csv(args.checks),
        layer_indices=args.layer_indices,
    )
    missing = validate_model_health_options(args.suite, options)
    if missing:
        raise ValueError(f"Missing required model health fields: {', '.join(missing)}")
    records = _load_records(client, options.dataset_name)
    report = run_model_health_suite(args.suite, options, records)
    print(json.dumps(report, indent=2))
    return 0


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _load_records(client: CrucibleClient, dataset_name: str) -> list[object]:
    if not dataset_name:
        return []
    _, records = client.dataset(dataset_name).load_records()
    return list(records)
