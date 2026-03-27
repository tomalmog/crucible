"""CLI commands for dataset operations: ingest, export-training.

Extracted from main.py to keep the dispatcher under the 300-line limit.
"""

from __future__ import annotations

import argparse

from core.constants import DEFAULT_QUALITY_MODEL
from core.types import IngestOptions
from store.dataset_sdk import CrucibleClient
from transforms.quality_scoring import supported_quality_models


def run_ingest_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle ingest command."""
    options = IngestOptions(
        dataset_name=args.dataset,
        source_uri=args.source,
        resume=args.resume,
        quality_model=args.quality_model,
    )
    dataset_name = client.ingest(options)
    print(dataset_name)
    return 0


def run_export_training_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle export-training command."""
    dataset = client.dataset(args.dataset)
    manifest_path = dataset.export_training(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        include_metadata=args.include_metadata,
    )
    print(manifest_path)
    return 0


def add_ingest_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ingest subcommand."""
    parser = subparsers.add_parser("ingest", help="Ingest a local path or S3 prefix")
    parser.add_argument("source", help="Source file, directory, or s3://bucket/prefix")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--resume", action="store_true", help="Resume from ingest checkpoint")
    parser.add_argument(
        "--quality-model", default=DEFAULT_QUALITY_MODEL,
        choices=supported_quality_models(), help="Quality scoring model",
    )


def add_dataset_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register dataset subcommand with list and delete actions."""
    parser = subparsers.add_parser("dataset", help="Dataset operations")
    sub = parser.add_subparsers(dest="dataset_action", required=True)
    sub.add_parser("list", help="List local datasets.")
    delete = sub.add_parser("delete", help="Delete a local dataset.")
    delete.add_argument("--name", required=True, help="Dataset name to delete")


def run_dataset_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Dispatch dataset subcommands."""
    action = args.dataset_action
    if action == "list":
        return _handle_dataset_list(client)
    if action == "delete":
        return _handle_dataset_delete(client, args.name)
    return 1


def _handle_dataset_list(client: CrucibleClient) -> int:
    """List all locally ingested datasets."""
    from pathlib import Path

    datasets_dir = client._config.data_root / "datasets"
    if not datasets_dir.is_dir():
        print("No datasets.")
        return 0
    names = sorted(d.name for d in datasets_dir.iterdir() if d.is_dir())
    if not names:
        print("No datasets.")
        return 0
    for name in names:
        manifest = datasets_dir / name / "manifest.json"
        if manifest.exists():
            import json
            data = json.loads(manifest.read_text())
            count = data.get("record_count", "?")
            source = data.get("source_uri", "")
            print(f"  {name}  ({count} records)  {source}")
        else:
            print(f"  {name}")
    return 0


def _handle_dataset_delete(client: CrucibleClient, name: str) -> int:
    """Delete a local dataset."""
    import shutil
    from pathlib import Path

    dataset_dir = client._config.data_root / "datasets" / name
    if not dataset_dir.is_dir():
        print(f"Dataset '{name}' not found.")
        return 1
    shutil.rmtree(dataset_dir)
    print(f"Deleted dataset '{name}'.")
    return 0


def add_export_training_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register export-training subcommand."""
    parser = subparsers.add_parser(
        "export-training", help="Export dataset into sharded local training files",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--shard-size", type=int, default=1000, help="Records per shard file")
    parser.add_argument(
        "--include-metadata", action="store_true",
        help="Include metadata fields in each shard row",
    )
