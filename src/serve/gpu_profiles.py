"""Pre-built GPU profiles for hardware-aware configuration.

This module provides recommended training configurations for
common GPU hardware including consumer and data center cards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuProfile:
    """Hardware profile for a specific GPU model.

    Attributes:
        name: GPU model name.
        vram_gb: Video memory in GB.
        compute_capability: CUDA compute capability.
        fp16_tflops: Half-precision throughput.
        recommended_batch_size: Suggested batch size for typical workloads.
        recommended_precision: Suggested precision mode.
        supports_flash_attention: Whether Flash Attention is supported.
        tdp_watts: Thermal design power in watts.
    """

    name: str
    vram_gb: float
    compute_capability: str
    fp16_tflops: float
    recommended_batch_size: int
    recommended_precision: str
    supports_flash_attention: bool
    tdp_watts: int


GPU_PROFILES: dict[str, GpuProfile] = {
    "rtx4090": GpuProfile(
        name="NVIDIA RTX 4090",
        vram_gb=24.0, compute_capability="8.9", fp16_tflops=82.6,
        recommended_batch_size=8, recommended_precision="bf16",
        supports_flash_attention=True, tdp_watts=450,
    ),
    "rtx4080": GpuProfile(
        name="NVIDIA RTX 4080",
        vram_gb=16.0, compute_capability="8.9", fp16_tflops=48.7,
        recommended_batch_size=4, recommended_precision="bf16",
        supports_flash_attention=True, tdp_watts=320,
    ),
    "rtx3090": GpuProfile(
        name="NVIDIA RTX 3090",
        vram_gb=24.0, compute_capability="8.6", fp16_tflops=35.6,
        recommended_batch_size=8, recommended_precision="fp16",
        supports_flash_attention=True, tdp_watts=350,
    ),
    "a100_40gb": GpuProfile(
        name="NVIDIA A100 40GB",
        vram_gb=40.0, compute_capability="8.0", fp16_tflops=77.97,
        recommended_batch_size=16, recommended_precision="bf16",
        supports_flash_attention=True, tdp_watts=400,
    ),
    "a100_80gb": GpuProfile(
        name="NVIDIA A100 80GB",
        vram_gb=80.0, compute_capability="8.0", fp16_tflops=77.97,
        recommended_batch_size=32, recommended_precision="bf16",
        supports_flash_attention=True, tdp_watts=400,
    ),
    "h100": GpuProfile(
        name="NVIDIA H100",
        vram_gb=80.0, compute_capability="9.0", fp16_tflops=267.6,
        recommended_batch_size=64, recommended_precision="bf16",
        supports_flash_attention=True, tdp_watts=700,
    ),
    "m1": GpuProfile(
        name="Apple M1",
        vram_gb=16.0, compute_capability="mps", fp16_tflops=5.5,
        recommended_batch_size=4, recommended_precision="fp32",
        supports_flash_attention=False, tdp_watts=20,
    ),
    "m2": GpuProfile(
        name="Apple M2",
        vram_gb=24.0, compute_capability="mps", fp16_tflops=7.1,
        recommended_batch_size=8, recommended_precision="fp32",
        supports_flash_attention=False, tdp_watts=22,
    ),
    "m3_max": GpuProfile(
        name="Apple M3 Max",
        vram_gb=48.0, compute_capability="mps", fp16_tflops=14.2,
        recommended_batch_size=16, recommended_precision="fp32",
        supports_flash_attention=False, tdp_watts=40,
    ),
    "m4_max": GpuProfile(
        name="Apple M4 Max",
        vram_gb=64.0, compute_capability="mps", fp16_tflops=18.0,
        recommended_batch_size=16, recommended_precision="fp32",
        supports_flash_attention=False, tdp_watts=45,
    ),
}


def get_gpu_profile(gpu_name: str) -> GpuProfile | None:
    """Look up a GPU profile by name."""
    key = gpu_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for profile_key, profile in GPU_PROFILES.items():
        normalized = profile_key.lower().replace("_", "")
        if key == normalized or key in profile.name.lower().replace(" ", ""):
            return profile
    return None


def list_gpu_profiles() -> list[GpuProfile]:
    """List all known GPU profiles."""
    return list(GPU_PROFILES.values())
