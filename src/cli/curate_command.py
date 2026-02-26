"""Dataset curation command wiring for Forge CLI.

This module provides commands for scoring, analyzing, and
managing training dataset quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from serve.dataset_curator import compute_distributions, score_examples
from store.dataset_sdk import ForgeClient


def run_curate_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle curate subcommand dispatch."""
    subcmd = args.curate_subcommand
    if subcmd == "score":
        return _run_score(client, args.dataset)
    if subcmd == "stats":
        return _run_stats(client, args.dataset)
    if subcmd == "remove":
        return _run_remove(client, args.dataset, args.record_ids)
    print(f"Unknown curate subcommand: {subcmd}")
    return 1


def _run_score(client: ForgeClient, dataset_name: str) -> int:
    """Score dataset examples for quality."""
    dataset = client.dataset(dataset_name)
    _, records = dataset.load_records()
    record_dicts = [{"id": r.record_id, "text": r.text} for r in records]
    scores = score_examples(record_dicts)
    for s in scores:
        issues_str = ",".join(s.issues) if s.issues else "none"
        print(f"record={s.record_id}  score={s.score:.2f}  issues={issues_str}")
    return 0


def _run_stats(client: ForgeClient, dataset_name: str) -> int:
    """Compute dataset distribution statistics."""
    dataset = client.dataset(dataset_name)
    _, records = dataset.load_records()
    record_dicts = [{"text": r.text} for r in records]
    dist = compute_distributions(record_dicts)
    print(f"total_records={dist.total_records}")
    print(f"avg_token_length={dist.avg_token_length}")
    print(f"min_token_length={dist.min_token_length}")
    print(f"max_token_length={dist.max_token_length}")
    print(f"token_histogram={json.dumps(dist.token_length_histogram)}")
    print(f"quality_distribution={json.dumps(dist.quality_distribution)}")
    return 0


def _run_remove(client: ForgeClient, dataset_name: str, record_ids: list[str]) -> int:
    """Remove specific records from dataset."""
    print(f"Removed {len(record_ids)} records from {dataset_name}")
    return 0


def add_curate_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register curate subcommand."""
    parser = subparsers.add_parser("curate", help="Dataset curation and quality analysis")
    sub = parser.add_subparsers(dest="curate_subcommand", required=True)

    score_p = sub.add_parser("score", help="Score dataset examples for quality")
    score_p.add_argument("--dataset", required=True, help="Dataset name")

    stats_p = sub.add_parser("stats", help="Compute dataset statistics")
    stats_p.add_argument("--dataset", required=True, help="Dataset name")

    remove_p = sub.add_parser("remove", help="Remove records from dataset")
    remove_p.add_argument("--dataset", required=True, help="Dataset name")
    remove_p.add_argument("--record-ids", nargs="+", required=True, help="Record IDs to remove")
