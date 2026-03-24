"""HuggingFace Hub command wiring for Crucible CLI.

This module provides commands for searching, downloading, and
pushing models and datasets to the HuggingFace Hub.
"""

from __future__ import annotations

import argparse
import sys

from serve.huggingface_hub import (
    download_dataset,
    download_model,
    get_dataset_info,
    get_model_info,
    search_datasets,
    search_models,
)
from store.dataset_sdk import CrucibleClient


def run_hub_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle hub subcommand dispatch."""
    subcmd = args.hub_subcommand
    if subcmd == "search-models":
        return _run_search_models(
            args.query, args.limit, getattr(args, "json", False),
            author=getattr(args, "author", ""),
            library=getattr(args, "library", ""),
            filter_tag=getattr(args, "filter", ""),
            sort=getattr(args, "sort", "downloads"),
        )
    if subcmd == "model-info":
        return _run_model_info(args.repo_id, getattr(args, "json", False))
    if subcmd == "download-model":
        return _run_download_model(
            client, args.repo_id, args.target_dir, args.revision,
            register=getattr(args, "register", False),
            model_name=getattr(args, "model_name", "") or "",
        )
    if subcmd == "download-model-remote":
        return _run_download_model_remote(
            client, args.repo_id, args.cluster,
            getattr(args, "model_name", "") or "",
            getattr(args, "revision", None),
            register=getattr(args, "register", False),
        )
    if subcmd == "search-datasets":
        return _run_search_datasets(
            args.query, args.limit, getattr(args, "json", False),
            author=getattr(args, "author", ""),
            filter_tag=getattr(args, "filter", ""),
            sort=getattr(args, "sort", "downloads"),
        )
    if subcmd == "dataset-info":
        return _run_dataset_info(args.repo_id, getattr(args, "json", False))
    if subcmd == "download-dataset":
        return _run_download_dataset(
            client, args.repo_id, args.target_dir, args.revision,
            register=getattr(args, "register", False),
            dataset_name=getattr(args, "dataset_name", "") or "",
        )
    if subcmd == "download-dataset-remote":
        return _run_download_dataset_remote(
            client, args.repo_id, args.cluster, getattr(args, "revision", None),
        )
    print(f"Unknown hub subcommand: {subcmd}")
    return 1


def _run_search_models(
    query: str, limit: int, json_output: bool = False,
    author: str = "", library: str = "", filter_tag: str = "", sort: str = "downloads",
) -> int:
    """Search for models on HuggingFace Hub."""
    import json as _json
    tags = [filter_tag] if filter_tag else None
    models = search_models(query, limit, author=author, filter_tags=tags, library=library, sort=sort)
    if not models:
        if json_output:
            print("[]")
        else:
            print("No models found.")
        return 0
    if json_output:
        print(_json.dumps([
            {"repo_id": m.repo_id, "author": m.author, "downloads": m.downloads,
             "likes": m.likes, "tags": list(m.tags), "task": m.pipeline_tag,
             "last_modified": m.last_modified, "total_size": m.total_size}
            for m in models
        ]))
    else:
        for m in models:
            print(f"{m.repo_id}  downloads={m.downloads}  task={m.pipeline_tag}")
    return 0


def _run_model_info(repo_id: str, json_output: bool = False) -> int:
    """Fetch detailed model info from HuggingFace Hub."""
    import json as _json
    info = get_model_info(repo_id)
    if json_output:
        print(_json.dumps(info))
    else:
        print(f"{info['repo_id']}  size={info['total_size']}  license={info['license']}")
    return 0


def _run_download_model(
    client: CrucibleClient, repo_id: str, target_dir: str, revision: str | None,
    *, register: bool = False, model_name: str = "",
) -> int:
    """Download a model from HuggingFace Hub."""
    path = download_model(repo_id, target_dir, revision)
    print(f"model_path={path}")
    if register:
        try:
            registry = client.model_registry()
            entry = registry.register_model(model_name or repo_id, path)
            print(f"model_name={entry.model_name}")
        except Exception as exc:
            print(f"warning: registration failed: {exc}", file=sys.stderr)
    return 0


def _run_download_model_remote(
    client: CrucibleClient, repo_id: str, cluster: str,
    model_name: str, revision: str | None, *, register: bool = False,
) -> int:
    """Download a model to a remote cluster."""
    from serve.remote_hub_download import download_model_to_cluster
    download_model_to_cluster(
        data_root=client._config.data_root, repo_id=repo_id,
        cluster_name=cluster, model_name=model_name,
        revision=revision, register=register,
    )
    return 0


def _run_search_datasets(
    query: str, limit: int, json_output: bool = False,
    author: str = "", filter_tag: str = "", sort: str = "downloads",
) -> int:
    """Search for datasets on HuggingFace Hub."""
    import json as _json
    tags = [filter_tag] if filter_tag else None
    datasets = search_datasets(query, limit, author=author, filter_tags=tags, sort=sort)
    if not datasets:
        if json_output:
            print("[]")
        else:
            print("No datasets found.")
        return 0
    if json_output:
        print(_json.dumps([
            {"repo_id": d.repo_id, "author": d.author, "downloads": d.downloads,
             "tags": list(d.tags), "last_modified": d.last_modified,
             "total_size": d.total_size}
            for d in datasets
        ]))
    else:
        for d in datasets:
            print(f"{d.repo_id}  downloads={d.downloads}")
    return 0


def _run_dataset_info(repo_id: str, json_output: bool = False) -> int:
    """Fetch detailed dataset info from HuggingFace Hub."""
    import json as _json
    info = get_dataset_info(repo_id)
    if json_output:
        print(_json.dumps(info))
    else:
        print(f"{info['repo_id']}  size={info['total_size']}  license={info['license']}")
    return 0


def _run_download_dataset(
    client: CrucibleClient, repo_id: str, target_dir: str, revision: str | None,
    *, register: bool = False, dataset_name: str = "",
) -> int:
    """Download a dataset from HuggingFace Hub."""
    path = download_dataset(repo_id, target_dir, revision)
    print(f"dataset_path={path}")
    if register:
        _register_downloaded_dataset(client, path, dataset_name or repo_id)
    return 0


def _run_download_dataset_remote(
    client: CrucibleClient, repo_id: str, cluster: str, revision: str | None,
) -> int:
    """Download a dataset to a remote cluster.

    Registration is automatic — download_dataset_to_cluster writes
    metadata.json so list_remote_datasets discovers the dataset.
    """
    from serve.remote_hub_download import download_dataset_to_cluster
    download_dataset_to_cluster(
        data_root=client._config.data_root, repo_id=repo_id,
        cluster_name=cluster, revision=revision,
    )
    return 0


def _register_downloaded_dataset(
    client: CrucibleClient, source_path: str, dataset_name: str,
) -> None:
    """Ingest a downloaded dataset into the Crucible dataset registry."""
    from core.types import IngestOptions
    try:
        name = client.ingest(IngestOptions(dataset_name=dataset_name, source_uri=source_path))
        print(f"dataset_registered={name}")
    except Exception as exc:
        print(f"warning: dataset registration failed: {exc}", file=sys.stderr)


def _add_register_args(parser: argparse.ArgumentParser, kind: str) -> None:
    """Add --register and --<kind>-name flags to a download subparser."""
    registry = "model registry" if kind == "model" else "dataset registry"
    parser.add_argument("--register", action="store_true", help=f"Register in {registry}")
    parser.add_argument(f"--{kind}-name", default="", help="Registry name (defaults to repo_id)")


def add_hub_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register hub subcommand."""
    parser = subparsers.add_parser("hub", help="HuggingFace Hub operations")
    sub = parser.add_subparsers(dest="hub_subcommand", required=True)

    sm = sub.add_parser("search-models", help="Search Hub models")
    sm.add_argument("query", nargs="?", default="", help="Search query")
    sm.add_argument("--limit", type=int, default=20, help="Max results")
    sm.add_argument("--json", action="store_true", help="Output as JSON")
    sm.add_argument("--author", default="", help="Filter by author")
    sm.add_argument("--library", default="", help="Filter by library (transformers, gguf, etc.)")
    sm.add_argument("--filter", default="", help="Filter by tag (text-generation, etc.)")
    sm.add_argument("--sort", default="downloads", help="Sort by: downloads, likes, createdAt")

    mi = sub.add_parser("model-info", help="Get detailed model info")
    mi.add_argument("repo_id", help="Model repository ID")
    mi.add_argument("--json", action="store_true", help="Output as JSON")

    dm = sub.add_parser("download-model", help="Download a model")
    dm.add_argument("repo_id", help="Model repository ID")
    dm.add_argument("--target-dir", default="./models", help="Download target directory")
    dm.add_argument("--revision", help="Model revision/branch")
    _add_register_args(dm, "model")

    dmr = sub.add_parser("download-model-remote", help="Download a model to a remote cluster")
    dmr.add_argument("repo_id", help="Model repository ID")
    dmr.add_argument("--cluster", required=True, help="Cluster name")
    dmr.add_argument("--revision", help="Model revision/branch")
    _add_register_args(dmr, "model")

    sd = sub.add_parser("search-datasets", help="Search Hub datasets")
    sd.add_argument("query", nargs="?", default="", help="Search query")
    sd.add_argument("--limit", type=int, default=20, help="Max results")
    sd.add_argument("--json", action="store_true", help="Output as JSON")
    sd.add_argument("--author", default="", help="Filter by author")
    sd.add_argument("--filter", default="", help="Filter by tag")
    sd.add_argument("--sort", default="downloads", help="Sort by: downloads, likes, createdAt")

    di = sub.add_parser("dataset-info", help="Get detailed dataset info")
    di.add_argument("repo_id", help="Dataset repository ID")
    di.add_argument("--json", action="store_true", help="Output as JSON")

    dd = sub.add_parser("download-dataset", help="Download a dataset")
    dd.add_argument("repo_id", help="Dataset repository ID")
    dd.add_argument("--target-dir", default="./datasets", help="Download target directory")
    dd.add_argument("--revision", help="Dataset revision/branch")
    _add_register_args(dd, "dataset")

    ddr = sub.add_parser("download-dataset-remote", help="Download a dataset to a remote cluster")
    ddr.add_argument("repo_id", help="Dataset repository ID")
    ddr.add_argument("--cluster", required=True, help="Cluster name")
    ddr.add_argument("--revision", help="Dataset revision/branch")
