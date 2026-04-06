"""Mixed-precision runtime helpers for training.

This module resolves requested precision mode against available hardware
and exposes autocast/scaler runtime objects for batch execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logging_config import get_logger
from core.types import PrecisionMode

_LOGGER = get_logger(__name__)


@dataclass
class TrainingPrecisionRuntime:
    """Resolved mixed-precision runtime state."""

    requested_mode: PrecisionMode
    resolved_mode: PrecisionMode
    autocast_enabled: bool
    autocast_dtype: Any | None
    scaler: Any | None


def build_training_precision_runtime(
    torch_module: Any,
    requested_mode: PrecisionMode,
    device: Any,
) -> TrainingPrecisionRuntime:
    """Resolve and construct mixed-precision runtime objects."""
    device_type = _resolve_device_type(device)
    resolved_mode = _resolve_precision_mode(torch_module, requested_mode, device_type)
    autocast_dtype = _resolve_autocast_dtype(torch_module, resolved_mode)
    scaler = _build_grad_scaler(torch_module, resolved_mode, device_type)
    runtime = TrainingPrecisionRuntime(
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        autocast_enabled=autocast_dtype is not None,
        autocast_dtype=autocast_dtype,
        scaler=scaler,
    )
    _LOGGER.info(
        "training_precision_resolved",
        requested_mode=requested_mode,
        resolved_mode=runtime.resolved_mode,
        device_type=device_type,
        autocast_enabled=runtime.autocast_enabled,
        grad_scaler_enabled=runtime.scaler is not None,
    )
    return runtime


def _resolve_device_type(device: Any) -> str:
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    return str(device)


def _resolve_precision_mode(
    torch_module: Any,
    requested_mode: PrecisionMode,
    device_type: str,
) -> PrecisionMode:
    if requested_mode == "fp32":
        return "fp32"
    if device_type != "cuda":
        if requested_mode != "auto":
            _LOGGER.warning(
                "training_precision_fallback",
                requested_mode=requested_mode,
                fallback_mode="fp32",
                reason="non_cuda_device",
            )
        return "fp32"
    if requested_mode == "fp16":
        return "fp16"
    if requested_mode == "bf16":
        if _is_bf16_supported(torch_module):
            return "bf16"
        _LOGGER.warning(
            "training_precision_fallback",
            requested_mode=requested_mode,
            fallback_mode="fp16",
            reason="bf16_not_supported",
        )
        return "fp16"
    return "bf16" if _is_bf16_supported(torch_module) else "fp16"


def _is_bf16_supported(torch_module: Any) -> bool:
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return False
    # Require compute capability >= 8.0 (Ampere+) for reliable bf16.
    # Older GPUs (Maxwell, Pascal, Volta, Turing) may report bf16 support
    # via is_bf16_supported() but lack the SDPA kernels needed for training.
    try:
        major, _ = cuda_module.get_device_capability(0)
        if major < 8:
            return False
    except Exception:
        pass
    bf16_probe = getattr(cuda_module, "is_bf16_supported", None)
    if callable(bf16_probe):
        return bool(bf16_probe())
    return False


def _resolve_autocast_dtype(torch_module: Any, resolved_mode: PrecisionMode) -> Any | None:
    if resolved_mode == "fp16":
        return torch_module.float16
    if resolved_mode == "bf16":
        return torch_module.bfloat16
    return None


def _build_grad_scaler(
    torch_module: Any,
    resolved_mode: PrecisionMode,
    device_type: str,
) -> Any | None:
    if resolved_mode != "fp16" or device_type != "cuda":
        return None
    amp_module = getattr(getattr(torch_module, "cuda", None), "amp", None)
    if amp_module is None:
        return None
    grad_scaler_cls = getattr(amp_module, "GradScaler", None)
    if grad_scaler_cls is None:
        return None
    return grad_scaler_cls(enabled=True)
