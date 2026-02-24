"""Hardware capability detection for training defaults.

This module inspects available accelerators and emits a stable profile used
for runtime recommendations and capacity-aware UI/CLI hints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.types import PrecisionMode
from serve.device_selection import is_mps_available, is_tpu_available


@dataclass(frozen=True)
class GpuHardware:
    """One detected GPU device summary."""

    index: int
    name: str
    total_memory_gb: float
    capability: str | None


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware capabilities and recommended defaults."""

    accelerator: str
    gpu_count: int
    gpus: tuple[GpuHardware, ...]
    bf16_supported: bool
    recommended_precision_mode: PrecisionMode
    recommended_batch_size: int
    suggested_profile: str

    def to_dict(self) -> dict[str, object]:
        """Serialize profile into a stable JSON-friendly mapping."""
        payload = asdict(self)
        payload["gpus"] = [asdict(gpu) for gpu in self.gpus]
        return payload


def detect_hardware_profile(torch_module: Any | None = None) -> HardwareProfile:
    """Detect local hardware profile and return recommended run defaults."""
    if is_tpu_available():
        return _tpu_profile()
    runtime_torch = torch_module or _import_torch_optional()
    if runtime_torch is None:
        return _cpu_profile()
    cuda_module = getattr(runtime_torch, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        gpu_count = int(cuda_module.device_count())
        gpus = tuple(_read_gpu_hardware(cuda_module, index) for index in range(gpu_count))
        bf16_supported = _read_bf16_support(cuda_module)
        precision_mode: PrecisionMode = "bf16" if bf16_supported else "fp16"
        suggested_profile = _suggest_profile(gpus)
        batch_size = _recommend_batch_size(gpus)
        return HardwareProfile(
            accelerator="cuda",
            gpu_count=gpu_count,
            gpus=gpus,
            bf16_supported=bf16_supported,
            recommended_precision_mode=precision_mode,
            recommended_batch_size=batch_size,
            suggested_profile=suggested_profile,
        )
    if is_mps_available(runtime_torch):
        return _mps_profile()
    return _cpu_profile()


def _import_torch_optional() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        accelerator="cpu",
        gpu_count=0,
        gpus=(),
        bf16_supported=False,
        recommended_precision_mode="fp32",
        recommended_batch_size=4,
        suggested_profile="cpu",
    )


def _tpu_profile() -> HardwareProfile:
    from serve.tpu_setup import import_xla
    xla = import_xla()
    try:
        num_devices = int(xla.xrt_world_size())
    except Exception:
        num_devices = 1
    return HardwareProfile(
        accelerator="tpu",
        gpu_count=num_devices,
        gpus=(),
        bf16_supported=True,
        recommended_precision_mode="bf16",
        recommended_batch_size=32,
        suggested_profile="tpu",
    )


def _mps_profile() -> HardwareProfile:
    return HardwareProfile(
        accelerator="mps",
        gpu_count=1,
        gpus=(
            GpuHardware(
                index=0,
                name="Apple MPS",
                total_memory_gb=0.0,
                capability=None,
            ),
        ),
        bf16_supported=False,
        recommended_precision_mode="fp32",
        recommended_batch_size=8,
        suggested_profile="apple_mps",
    )


def _read_gpu_hardware(cuda_module: Any, index: int) -> GpuHardware:
    device_props = cuda_module.get_device_properties(index)
    total_memory = float(getattr(device_props, "total_memory", 0.0))
    total_memory_gb = round(total_memory / (1024**3), 2)
    major = getattr(device_props, "major", None)
    minor = getattr(device_props, "minor", None)
    capability = None
    if isinstance(major, int) and isinstance(minor, int):
        capability = f"{major}.{minor}"
    name = str(getattr(device_props, "name", f"gpu-{index}"))
    return GpuHardware(
        index=index,
        name=name,
        total_memory_gb=total_memory_gb,
        capability=capability,
    )


def _read_bf16_support(cuda_module: Any) -> bool:
    support_probe = getattr(cuda_module, "is_bf16_supported", None)
    if callable(support_probe):
        return bool(support_probe())
    return False


def _suggest_profile(gpus: tuple[GpuHardware, ...]) -> str:
    gpu_names = " ".join(gpu.name.upper() for gpu in gpus)
    if "H100" in gpu_names:
        return "h100"
    if "A100" in gpu_names:
        return "a100"
    if "L40" in gpu_names:
        return "l40"
    return "generic_cuda"


def _recommend_batch_size(gpus: tuple[GpuHardware, ...]) -> int:
    if not gpus:
        return 4
    min_memory_gb = min(gpu.total_memory_gb for gpu in gpus)
    if min_memory_gb >= 80:
        return 64
    if min_memory_gb >= 48:
        return 40
    if min_memory_gb >= 24:
        return 24
    if min_memory_gb >= 16:
        return 16
    return 8
