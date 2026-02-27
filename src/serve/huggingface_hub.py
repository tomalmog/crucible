"""HuggingFace Hub operations for model and dataset management.

This module provides search, download, and upload capabilities
for models and datasets on the HuggingFace Hub.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError


@dataclass(frozen=True)
class HubModelInfo:
    """Information about a model on HuggingFace Hub.

    Attributes:
        repo_id: Repository identifier (e.g. 'meta-llama/Llama-2-7b').
        author: Model author/organization.
        downloads: Total download count.
        likes: Number of likes.
        tags: Model tags.
        pipeline_tag: Primary task tag.
        last_modified: ISO timestamp of last modification.
        total_size: Total size in bytes (0 when unavailable).
    """

    repo_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()
    pipeline_tag: str = ""
    last_modified: str = ""
    total_size: int = 0


@dataclass(frozen=True)
class HubDatasetInfo:
    """Information about a dataset on HuggingFace Hub.

    Attributes:
        repo_id: Repository identifier.
        author: Dataset author/organization.
        downloads: Total download count.
        tags: Dataset tags.
        last_modified: ISO timestamp of last modification.
        total_size: Total size in bytes (0 when unavailable).
    """

    repo_id: str
    author: str = ""
    downloads: int = 0
    tags: tuple[str, ...] = ()
    last_modified: str = ""
    total_size: int = 0


def _import_huggingface_hub() -> Any:
    """Import huggingface_hub library."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError as error:
        raise ForgeDependencyError(
            "HuggingFace Hub operations require the huggingface_hub package. "
            "Install it with: pip install huggingface_hub"
        ) from error


def search_models(
    query: str,
    limit: int = 20,
    author: str = "",
    filter_tags: list[str] | None = None,
    library: str = "",
    sort: str = "downloads",
) -> list[HubModelInfo]:
    """Search HuggingFace Hub for models matching a query."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    tags = list(filter_tags) if filter_tags else []
    if library:
        tags.append(library)
    kwargs: dict[str, Any] = {"search": query, "limit": limit, "sort": sort}
    if author:
        kwargs["author"] = author
    if tags:
        kwargs["filter"] = tags
    results = list(api.list_models(**kwargs))
    models: list[HubModelInfo] = []
    for model in results:
        modified = ""
        if hasattr(model, "last_modified") and model.last_modified:
            modified = model.last_modified.isoformat() if hasattr(model.last_modified, "isoformat") else str(model.last_modified)
        models.append(HubModelInfo(
            repo_id=model.id or "",
            author=model.author or "",
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            tags=tuple(model.tags or []),
            pipeline_tag=model.pipeline_tag or "",
            last_modified=modified,
        ))
    sizes = _fetch_model_sizes(api, [m.repo_id for m in models])
    return [
        HubModelInfo(
            repo_id=m.repo_id, author=m.author, downloads=m.downloads,
            likes=m.likes, tags=m.tags, pipeline_tag=m.pipeline_tag,
            last_modified=m.last_modified, total_size=sizes.get(m.repo_id, 0),
        )
        for m in models
    ]


def _fetch_repo_size(api: Any, repo_id: str, repo_type: str) -> tuple[str, int]:
    """Fetch total file size for a single repo. Returns (repo_id, size)."""
    try:
        if repo_type == "model":
            info = api.model_info(repo_id, files_metadata=True)
        else:
            info = api.dataset_info(repo_id, files_metadata=True)
        total = sum(getattr(s, "size", None) or 0 for s in (info.siblings or []))
        return (repo_id, total)
    except Exception:
        return (repo_id, 0)


def _fetch_model_sizes(api: Any, repo_ids: list[str]) -> dict[str, int]:
    """Fetch total sizes for a batch of model repos in parallel."""
    return _fetch_sizes_parallel(api, repo_ids, "model")


def _fetch_dataset_sizes(api: Any, repo_ids: list[str]) -> dict[str, int]:
    """Fetch total sizes for a batch of dataset repos in parallel."""
    return _fetch_sizes_parallel(api, repo_ids, "dataset")


def _fetch_sizes_parallel(api: Any, repo_ids: list[str], repo_type: str) -> dict[str, int]:
    """Fetch sizes for repos in parallel using a thread pool."""
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_repo_size, api, rid, repo_type): rid for rid in repo_ids}
        for future in as_completed(futures):
            repo_id, size = future.result()
            sizes[repo_id] = size
    return sizes


def download_model(
    repo_id: str,
    target_dir: str,
    revision: str | None = None,
) -> str:
    """Download a model from HuggingFace Hub."""
    hf = _import_huggingface_hub()
    path = hf.snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        revision=revision,
    )
    return str(path)


def search_datasets(
    query: str,
    limit: int = 20,
    author: str = "",
    filter_tags: list[str] | None = None,
    sort: str = "downloads",
) -> list[HubDatasetInfo]:
    """Search HuggingFace Hub for datasets matching a query."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    kwargs: dict[str, Any] = {"search": query, "limit": limit, "sort": sort}
    if author:
        kwargs["author"] = author
    if filter_tags:
        kwargs["filter"] = filter_tags
    results = list(api.list_datasets(**kwargs))
    datasets: list[HubDatasetInfo] = []
    for ds in results:
        modified = ""
        if hasattr(ds, "last_modified") and ds.last_modified:
            modified = ds.last_modified.isoformat() if hasattr(ds.last_modified, "isoformat") else str(ds.last_modified)
        datasets.append(HubDatasetInfo(
            repo_id=ds.id or "",
            author=ds.author or "",
            downloads=ds.downloads or 0,
            tags=tuple(ds.tags or []),
            last_modified=modified,
        ))
    sizes = _fetch_dataset_sizes(api, [d.repo_id for d in datasets])
    return [
        HubDatasetInfo(
            repo_id=d.repo_id, author=d.author, downloads=d.downloads,
            tags=d.tags, last_modified=d.last_modified,
            total_size=sizes.get(d.repo_id, 0),
        )
        for d in datasets
    ]


def download_dataset(
    repo_id: str,
    target_dir: str,
    revision: str | None = None,
) -> str:
    """Download a dataset from HuggingFace Hub."""
    hf = _import_huggingface_hub()
    path = hf.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        revision=revision,
    )
    return str(path)


def get_model_info(repo_id: str) -> dict[str, Any]:
    """Fetch detailed model info from HuggingFace Hub."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    files = []
    total_size = 0
    for s in info.siblings or []:
        size = getattr(s, "size", None) or 0
        files.append({"filename": s.rfilename, "size": size})
        total_size += size
    card = getattr(info, "card_data", None)
    return {
        "repo_id": info.id,
        "author": info.author or "",
        "downloads": info.downloads or 0,
        "likes": info.likes or 0,
        "tags": list(info.tags or []),
        "task": info.pipeline_tag or "",
        "last_modified": info.last_modified.isoformat() if info.last_modified else "",
        "created_at": info.created_at.isoformat() if info.created_at else "",
        "license": getattr(card, "license", "") or "" if card else "",
        "base_model": (getattr(card, "base_model", "") or "") if card else "",
        "library": info.library_name or "",
        "gated": info.gated if info.gated else False,
        "files": files,
        "total_size": total_size,
    }


def get_dataset_info(repo_id: str) -> dict[str, Any]:
    """Fetch detailed dataset info from HuggingFace Hub."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    files = []
    total_size = 0
    for s in info.siblings or []:
        size = getattr(s, "size", None) or 0
        files.append({"filename": s.rfilename, "size": size})
        total_size += size
    card = getattr(info, "card_data", None)
    return {
        "repo_id": info.id,
        "author": info.author or "",
        "downloads": info.downloads or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "tags": list(info.tags or []),
        "task_categories": (getattr(card, "task_categories", []) or []) if card else [],
        "last_modified": info.last_modified.isoformat() if info.last_modified else "",
        "created_at": info.created_at.isoformat() if info.created_at else "",
        "license": getattr(card, "license", "") or "" if card else "",
        "gated": info.gated if info.gated else False,
        "files": files,
        "total_size": total_size,
    }


def push_model(
    model_path: str,
    repo_id: str,
    commit_message: str = "Upload model via Forge",
    private: bool = False,
) -> str:
    """Push a trained model to HuggingFace Hub."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    url = api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    return str(url)
