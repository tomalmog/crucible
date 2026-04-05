"""CLI commands for custom benchmark management."""

from __future__ import annotations

import argparse

from store.dataset_sdk import CrucibleClient


def add_benchmark_registry_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("benchmark-registry", help="Custom benchmark operations")
    sub = parser.add_subparsers(dest="benchmark_action", required=True)

    create = sub.add_parser("create", help="Create a custom benchmark")
    create.add_argument("--name", required=True, help="Benchmark name")
    create.add_argument("--source", required=True, help="Path to JSONL file with prompt/response fields")

    list_cmd = sub.add_parser("list", help="List custom benchmarks")
    list_cmd.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    delete = sub.add_parser("delete", help="Delete a custom benchmark")
    delete.add_argument("--name", required=True, help="Benchmark name to delete")


def run_benchmark_registry_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    from eval.custom_benchmarks import (
        create_custom_benchmark,
        list_custom_benchmarks,
        delete_custom_benchmark,
    )

    action = args.benchmark_action
    data_root = client._config.data_root

    if action == "create":
        info = create_custom_benchmark(data_root, args.name, args.source)
        print(f"Created benchmark '{info.name}' with {info.entries} entries")
        return 0

    if action == "list":
        benchmarks = list_custom_benchmarks(data_root)
        if getattr(args, "json_output", False):
            import json
            print(json.dumps([
                {"name": b.name, "entries": b.entries, "created_at": b.created_at}
                for b in benchmarks
            ]))
        elif not benchmarks:
            print("No custom benchmarks.")
        else:
            for b in benchmarks:
                print(f"  {b.name}  ({b.entries} entries)  {b.created_at}")
        return 0

    if action == "delete":
        delete_custom_benchmark(data_root, args.name)
        print(f"Deleted benchmark '{args.name}'.")
        return 0

    return 1
