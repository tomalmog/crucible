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
