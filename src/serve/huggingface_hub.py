"""HuggingFace Hub operations for model and dataset management.

This module provides search, download, and upload capabilities
for models and datasets on the HuggingFace Hub.
"""

from __future__ import annotations

import json
import shutil
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
    """

    repo_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()
    pipeline_tag: str = ""
    last_modified: str = ""


@dataclass(frozen=True)
class HubDatasetInfo:
    """Information about a dataset on HuggingFace Hub.

    Attributes:
        repo_id: Repository identifier.
        author: Dataset author/organization.
        downloads: Total download count.
        tags: Dataset tags.
        last_modified: ISO timestamp of last modification.
    """

    repo_id: str
    author: str = ""
    downloads: int = 0
    tags: tuple[str, ...] = ()
    last_modified: str = ""


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


def search_models(query: str, limit: int = 20) -> list[HubModelInfo]:
    """Search HuggingFace Hub for models matching a query."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    results = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
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
    return models


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


def search_datasets(query: str, limit: int = 20) -> list[HubDatasetInfo]:
    """Search HuggingFace Hub for datasets matching a query."""
    hf = _import_huggingface_hub()
    api = hf.HfApi()
    results = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
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
    return datasets


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
