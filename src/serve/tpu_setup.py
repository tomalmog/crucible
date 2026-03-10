"""TPU device detection and XLA mesh initialization.

This module handles Google Cloud TPU configuration using torch_xla
as an optional dependency. It provides detection, device resolution,
and mesh initialization for TPU-based training workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleDistributedError


@dataclass(frozen=True)
class TpuDeviceInfo:
    """Detected TPU device summary."""

    device_type: str
    num_devices: int
    xla_device: Any | None


def detect_tpu_availability() -> bool:
    """Return True if torch_xla is installed and TPU devices are available."""
    xla_module = _try_import_xla()
    if xla_module is None:
        return False
    try:
        device = xla_module.xla_device()
        return device is not None
    except Exception:
        return False


def resolve_tpu_device(xla_module: Any) -> Any:
    """Resolve the XLA device for the current process.

    Args:
        xla_module: Imported torch_xla module.

    Returns:
        XLA device for the current process.

    Raises:
        CrucibleDistributedError: If no TPU device is available.
    """
    try:
        device = xla_module.xla_device()
    except Exception as error:
        raise CrucibleDistributedError(
            f"Failed to resolve TPU device: {error}. "
            "Check TPU configuration and torch_xla installation."
        ) from error
    if device is None:
        raise CrucibleDistributedError("No TPU device available.")
    return device


def get_tpu_device_info(xla_module: Any) -> TpuDeviceInfo:
    """Build a summary of detected TPU devices.

    Args:
        xla_module: Imported torch_xla module.

    Returns:
        TPU device info with device count and type.
    """
    device = resolve_tpu_device(xla_module)
    num_devices = _get_world_size(xla_module)
    return TpuDeviceInfo(
        device_type="tpu",
        num_devices=num_devices,
        xla_device=device,
    )


def init_xla_mesh(xla_module: Any) -> None:
    """Initialize XLA process group for multi-device TPU training.

    Args:
        xla_module: Imported torch_xla module.

    Raises:
        CrucibleDistributedError: If mesh initialization fails.
    """
    try:
        runtime = getattr(xla_module, "runtime", None)
        if runtime is not None and hasattr(runtime, "initialize_cache"):
            runtime.initialize_cache("/tmp/xla_cache", readonly=False)
    except Exception as error:
        raise CrucibleDistributedError(
            f"Failed to initialize XLA mesh: {error}."
        ) from error


def import_xla() -> Any:
    """Import torch_xla or raise CrucibleDependencyError."""
    xla = _try_import_xla()
    if xla is None:
        raise CrucibleDependencyError(
            "TPU support requires torch_xla. "
            "Install with 'pip install torch_xla'."
        )
    return xla


def _try_import_xla() -> Any | None:
    """Try to import torch_xla, return None if unavailable."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]
        return xm
    except ImportError:
        return None


def _get_world_size(xla_module: Any) -> int:
    """Return the number of XLA devices."""
    try:
        return int(xla_module.xrt_world_size())
    except Exception:
        return 1
