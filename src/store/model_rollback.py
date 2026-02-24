"""Model version rollback operations.

This module provides the ability to set a previously registered
model version as the current active version in the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import ForgeModelRegistryError
from core.model_registry_types import ModelVersion

if TYPE_CHECKING:
    from store.model_registry import ModelRegistry


def rollback_active_version(
    registry: "ModelRegistry",
    version_id: str,
) -> ModelVersion:
    """Set the active version in the registry to the specified version.

    Validates that the version exists, then updates the registry index
    to point the active version to the requested version_id.

    Args:
        registry: Model registry instance to update.
        version_id: Identifier of the version to roll back to.

    Returns:
        The ModelVersion that is now marked as active.

    Raises:
        ForgeModelRegistryError: If the version does not exist.
    """
    version = registry.get_version(version_id)
    _update_active_in_index(registry, version_id)
    return version


def _update_active_in_index(
    registry: "ModelRegistry",
    version_id: str,
) -> None:
    """Update the active_version_id in the registry index.

    Args:
        registry: Model registry instance to update.
        version_id: New active version identifier.
    """
    from store.model_registry_io import (
        load_registry_index,
        save_registry_index,
    )

    index = load_registry_index(registry.models_root)
    index["active_version_id"] = version_id
    save_registry_index(registry.models_root, index)
