"""Model version rollback operations.

This module provides the ability to set a previously registered
model version as the current active version within a model group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.model_registry_types import ModelVersion

if TYPE_CHECKING:
    from store.model_registry import ModelRegistry


def rollback_active_version(
    registry: "ModelRegistry",
    model_name: str,
    version_id: str,
) -> ModelVersion:
    """Set the active version in a model group to the specified version.

    Validates that the version exists, then updates the model group
    index to point the active version to the requested version_id.

    Args:
        registry: Model registry instance to update.
        model_name: Name of the model group to update.
        version_id: Identifier of the version to roll back to.

    Returns:
        The ModelVersion that is now marked as active.

    Raises:
        CrucibleModelRegistryError: If the version does not exist.
    """
    version = registry.get_version(version_id)
    _update_active_in_group(registry, model_name, version_id)
    return version


def _update_active_in_group(
    registry: "ModelRegistry",
    model_name: str,
    version_id: str,
) -> None:
    """Update the active_version_id in a model group index.

    Args:
        registry: Model registry instance to update.
        model_name: Name of the model group.
        version_id: New active version identifier.
    """
    from store.model_registry_io import (
        load_model_group,
        save_model_group,
    )

    group = load_model_group(registry.models_root, model_name)
    group["active_version_id"] = version_id
    save_model_group(registry.models_root, model_name, group)
