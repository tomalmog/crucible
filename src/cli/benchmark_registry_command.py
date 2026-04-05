"""CLI commands for benchmark registry management."""

from __future__ import annotations

import argparse

from store.dataset_sdk import CrucibleClient


def add_benchmark_registry_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("benchmark-registry", help="Benchmark registry operations")
    sub = parser.add_subparsers(dest="benchmark_action", required=True)

    create = sub.add_parser("create", help="Create a custom benchmark from JSONL")
    create.add_argument("--name", required=True, help="Benchmark name")
    create.add_argument("--source", required=True, help="Path to JSONL file")

    add = sub.add_parser("add", help="Add an lm-eval task to the registry")
    add.add_argument("--name", required=True, help="lm-eval task name")
    add.add_argument("--display-name", default="", help="Display name")
    add.add_argument("--description", default="", help="Description")

    list_cmd = sub.add_parser("list", help="List registered benchmarks")
    list_cmd.add_argument("--json", action="store_true", dest="json_output")

    delete = sub.add_parser("delete", help="Delete a benchmark")
    delete.add_argument("--name", required=True, help="Benchmark name")

    samples = sub.add_parser("samples", help="Show sample entries")
    samples.add_argument("--name", required=True, help="Benchmark name")
    samples.add_argument("--offset", type=int, default=0)
    samples.add_argument("--limit", type=int, default=25)
    samples.add_argument("--builtin", action="store_true")

    resolve = sub.add_parser("resolve-count", help="Fetch entry count for an lm-eval task")
    resolve.add_argument("--name", required=True, help="Benchmark name")

    search = sub.add_parser("search", help="Search lm-eval tasks")
    search.add_argument("query", help="Search query")
    search.add_argument("--limit", type=int, default=20)


def run_benchmark_registry_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    import json
    from eval.custom_benchmarks import (
        add_lm_eval_benchmark,
        create_custom_benchmark,
        delete_benchmark,
        list_benchmarks,
        sample_custom_benchmark,
        search_lm_eval_tasks,
    )

    action = args.benchmark_action
    data_root = client._config.data_root

    if action == "create":
        info = create_custom_benchmark(data_root, args.name, args.source)
        print(f"Created benchmark '{info.name}' with {info.entries} entries")
        return 0

    if action == "add":
        info = add_lm_eval_benchmark(
            data_root, args.name,
            display_name=args.display_name,
            description=args.description,
        )
        print(f"Added benchmark '{info.name}'")
        # Resolve entry count in-process
        from eval.custom_benchmarks import resolve_entry_count
        count = resolve_entry_count(data_root, args.name)
        if count > 0:
            print(f"Resolved {count} entries")
        return 0

    if action == "list":
        benchmarks = list_benchmarks(data_root)
        if getattr(args, "json_output", False):
            print(json.dumps([
                {"name": b.name, "display_name": b.display_name, "type": b.type,
                 "entries": b.entries, "description": b.description,
                 "created_at": b.created_at}
                for b in benchmarks
            ]))
        elif not benchmarks:
            print("No benchmarks.")
        else:
            for b in benchmarks:
                print(f"  {b.display_name}  ({b.type}, {b.entries} entries)")
        return 0

    if action == "delete":
        delete_benchmark(data_root, args.name)
        print(f"Deleted benchmark '{args.name}'.")
        return 0

    if action == "samples":
        if getattr(args, "builtin", False):
            from eval.builtin_benchmark_samples import sample_builtin_benchmark
            result = sample_builtin_benchmark(args.name, args.offset, args.limit)
            print(json.dumps(result))
        else:
            rows = sample_custom_benchmark(data_root, args.name, args.offset, args.limit)
            print(json.dumps(rows))
        return 0

    if action == "resolve-count":
        from eval.custom_benchmarks import resolve_entry_count
        count = resolve_entry_count(data_root, args.name)
        print(json.dumps({"name": args.name, "entries": count}))
        return 0

    if action == "search":
        results = search_lm_eval_tasks(args.query, args.limit)
        print(json.dumps(results))
        return 0

    return 1
