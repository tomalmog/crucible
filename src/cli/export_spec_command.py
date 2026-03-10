"""Export-spec command wiring for Crucible CLI.

This module exports a training run's configuration as a YAML run-spec
file, enabling reproducible pipeline execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from core.errors import CrucibleServeError
from core.run_spec_export import export_run_to_yaml
from core.types import TrainingOptions
from serve.training_run_types import TrainingRunRecord
from store.dataset_sdk import CrucibleClient


def run_export_spec_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle export-spec command invocation.

    Loads a training run's metadata and exports it as YAML.
    """
    run_record = client.get_training_run(args.run_id)
    config_path = _find_training_config(run_record)
    if config_path is None:
        print(f"Error: No training config found for run {args.run_id}.")
        return 1
    from serve.training_metadata import load_training_config

    config_dict = load_training_config(config_path)
    if config_dict is None:
        print(f"Error: Could not load training config for run {args.run_id}.")
        return 1
    options = TrainingOptions(**config_dict)
    yaml_content = export_run_to_yaml(options, run_id=args.run_id)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(yaml_content, encoding="utf-8")
        print(f"Exported run-spec to {output_path}")
    else:
        print(yaml_content)
    return 0


def add_export_spec_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register export-spec subcommand."""
    parser = subparsers.add_parser(
        "export-spec",
        help="Export a training run as a YAML run-spec file",
    )
    parser.add_argument("--run-id", required=True, help="Training run ID to export")
    parser.add_argument("--output", help="Output YAML file path (prints to stdout if omitted)")


def _find_training_config(run_record: TrainingRunRecord) -> str | None:
    """Locate the training config JSON beside the model path."""
    output_dir = run_record.output_dir
    if not output_dir:
        return None
    model_path = Path(output_dir) / "model.pt"
    if model_path.exists():
        return str(model_path)
    return None
