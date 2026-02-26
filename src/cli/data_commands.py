"""CLI commands for dataset operations: ingest, versions, filter, export-training.

Extracted from main.py to keep the dispatcher under the 300-line limit.
"""

from __future__ import annotations

import argparse

from core.constants import DEFAULT_QUALITY_MODEL
from core.types import IngestOptions, MetadataFilter
from store.dataset_sdk import ForgeClient
from transforms.quality_scoring import supported_quality_models


def run_ingest_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle ingest command."""
    options = IngestOptions(
        dataset_name=args.dataset,
        source_uri=args.source,
        output_uri=args.output_uri,
        resume=args.resume,
        incremental=args.incremental,
        quality_model=args.quality_model,
    )
    version_id = client.ingest(options)
    print(version_id)
    return 0


def run_versions_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle versions command."""
    dataset = client.dataset(args.dataset)
    for manifest in dataset.list_versions():
        print(
            f"{manifest.version_id}\t{manifest.record_count}\t"
            f"{manifest.created_at.isoformat()}\t{manifest.parent_version or '-'}"
        )
    return 0


def run_filter_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle filter command."""
    filter_spec = MetadataFilter(
        language=args.language,
        min_quality_score=args.min_quality,
        source_prefix=args.source_prefix,
    )
    dataset = client.dataset(args.dataset)
    print(dataset.filter(filter_spec))
    return 0


def run_export_training_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle export-training command."""
    dataset = client.dataset(args.dataset)
    manifest_path = dataset.export_training(
        output_dir=args.output_dir,
        version_id=args.version_id,
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
    parser.add_argument("--output-uri", help="Optional s3:// export destination")
    parser.add_argument("--resume", action="store_true", help="Resume from ingest checkpoint")
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only transform new/changed records against latest version",
    )
    parser.add_argument(
        "--quality-model", default=DEFAULT_QUALITY_MODEL,
        choices=supported_quality_models(), help="Quality scoring model",
    )


def add_versions_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register versions subcommand."""
    parser = subparsers.add_parser("versions", help="List dataset versions")
    parser.add_argument("--dataset", required=True, help="Dataset name")


def add_filter_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register filter subcommand."""
    parser = subparsers.add_parser("filter", help="Create metadata-filtered snapshot")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--language", help="Language filter, e.g. en")
    parser.add_argument("--min-quality", type=float, help="Minimum quality score")
    parser.add_argument("--source-prefix", help="Source URI prefix filter")


def add_export_training_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register export-training subcommand."""
    parser = subparsers.add_parser(
        "export-training", help="Export a version into sharded local training files",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--version-id", help="Optional specific version id")
    parser.add_argument("--shard-size", type=int, default=1000, help="Records per shard file")
    parser.add_argument(
        "--include-metadata", action="store_true",
        help="Include metadata fields in each shard row",
    )
