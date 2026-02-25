"""HuggingFace Hub command wiring for Forge CLI.

This module provides commands for searching, downloading, and
pushing models and datasets to the HuggingFace Hub.
"""

from __future__ import annotations

import argparse
from typing import Any

from serve.huggingface_hub import (
    download_dataset,
    download_model,
    push_model,
    search_datasets,
    search_models,
)
from store.dataset_sdk import ForgeClient


def run_hub_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle hub subcommand dispatch."""
    subcmd = args.hub_subcommand
    if subcmd == "search-models":
        return _run_search_models(args.query, args.limit, getattr(args, "json", False))
    if subcmd == "download-model":
        return _run_download_model(args.repo_id, args.target_dir, args.revision)
    if subcmd == "search-datasets":
        return _run_search_datasets(args.query, args.limit, getattr(args, "json", False))
    if subcmd == "download-dataset":
        return _run_download_dataset(args.repo_id, args.target_dir, args.revision)
    if subcmd == "push":
        return _run_push(args.model_path, args.repo_id, args.message, args.private)
    print(f"Unknown hub subcommand: {subcmd}")
    return 1


def _run_search_models(query: str, limit: int, json_output: bool = False) -> int:
    """Search for models on HuggingFace Hub."""
    import json as _json
    models = search_models(query, limit)
    if not models:
        if json_output:
            print("[]")
        else:
            print("No models found.")
        return 0
    if json_output:
        print(_json.dumps([
            {"repo_id": m.repo_id, "author": m.author, "downloads": m.downloads,
             "likes": m.likes, "tags": list(m.tags), "task": m.pipeline_tag}
            for m in models
        ]))
    else:
        for m in models:
            print(f"{m.repo_id}  downloads={m.downloads}  task={m.pipeline_tag}")
    return 0


def _run_download_model(repo_id: str, target_dir: str, revision: str | None) -> int:
    """Download a model from HuggingFace Hub."""
    path = download_model(repo_id, target_dir, revision)
    print(f"model_path={path}")
    return 0


def _run_search_datasets(query: str, limit: int, json_output: bool = False) -> int:
    """Search for datasets on HuggingFace Hub."""
    import json as _json
    datasets = search_datasets(query, limit)
    if not datasets:
        if json_output:
            print("[]")
        else:
            print("No datasets found.")
        return 0
    if json_output:
        print(_json.dumps([
            {"repo_id": d.repo_id, "author": d.author, "downloads": d.downloads,
             "tags": list(d.tags)}
            for d in datasets
        ]))
    else:
        for d in datasets:
            print(f"{d.repo_id}  downloads={d.downloads}")
    return 0


def _run_download_dataset(repo_id: str, target_dir: str, revision: str | None) -> int:
    """Download a dataset from HuggingFace Hub."""
    path = download_dataset(repo_id, target_dir, revision)
    print(f"dataset_path={path}")
    return 0


def _run_push(model_path: str, repo_id: str, message: str, private: bool) -> int:
    """Push a model to HuggingFace Hub."""
    url = push_model(model_path, repo_id, message, private)
    print(f"url={url}")
    return 0


def add_hub_command(subparsers: Any) -> None:
    """Register hub subcommand."""
    parser = subparsers.add_parser("hub", help="HuggingFace Hub operations")
    sub = parser.add_subparsers(dest="hub_subcommand", required=True)

    sm = sub.add_parser("search-models", help="Search Hub models")
    sm.add_argument("query", help="Search query")
    sm.add_argument("--limit", type=int, default=20, help="Max results")
    sm.add_argument("--json", action="store_true", help="Output as JSON")

    dm = sub.add_parser("download-model", help="Download a model")
    dm.add_argument("repo_id", help="Model repository ID")
    dm.add_argument("--target-dir", default="./models", help="Download target directory")
    dm.add_argument("--revision", help="Model revision/branch")

    sd = sub.add_parser("search-datasets", help="Search Hub datasets")
    sd.add_argument("query", help="Search query")
    sd.add_argument("--limit", type=int, default=20, help="Max results")
    sd.add_argument("--json", action="store_true", help="Output as JSON")

    dd = sub.add_parser("download-dataset", help="Download a dataset")
    dd.add_argument("repo_id", help="Dataset repository ID")
    dd.add_argument("--target-dir", default="./datasets", help="Download target directory")
    dd.add_argument("--revision", help="Dataset revision/branch")

    push = sub.add_parser("push", help="Push model to Hub")
    push.add_argument("model_path", help="Local model path")
    push.add_argument("repo_id", help="Target repository ID")
    push.add_argument("--message", default="Upload model via Forge", help="Commit message")
    push.add_argument("--private", action="store_true", help="Create private repo")
