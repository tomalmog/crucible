"""Auto-registration of remote training models into the local registry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from store.cluster_registry import load_cluster

if TYPE_CHECKING:
    from core.slurm_types import RemoteJobRecord


def auto_register_remote_model(
    data_root: Path,
    record: RemoteJobRecord,
    model_path: str,
) -> None:
    """Register a completed remote model in the local model registry."""
    try:
        from store.model_registry import ModelRegistry

        cluster = load_cluster(data_root, record.cluster_name)
        registry = ModelRegistry(data_root)
        model_name = (
            record.model_name
            or f"remote-{record.training_method}-{record.job_id[:16]}"
        )
        registry.register_remote_model(
            model_name=model_name,
            remote_host=cluster.host,
            remote_path=model_path,
            run_id=record.job_id,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to auto-register model for job %s", record.job_id,
            exc_info=True,
        )


def is_model_registered(data_root: Path, record: RemoteJobRecord) -> bool:
    """Check if a remote job's model is already in the registry."""
    try:
        from store.model_registry import ModelRegistry

        registry = ModelRegistry(data_root)
        model_name = (
            record.model_name
            or f"remote-{record.training_method}-{record.job_id[:16]}"
        )
        versions = registry.list_versions_for_model(model_name)
        return any(v.run_id == record.job_id for v in versions)
    except Exception:
        return False
