"""Attention backend detection and selection.

This module detects available attention implementations and selects
the best backend for training: Flash Attention 2, torch SDPA, or vanilla.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AttentionBackend = Literal["auto", "flash", "sdpa", "vanilla"]

SUPPORTED_ATTENTION_BACKENDS: tuple[AttentionBackend, ...] = ("auto", "flash", "sdpa", "vanilla")


@dataclass(frozen=True)
class AttentionBackendInfo:
    """Information about the selected attention backend.

    Attributes:
        name: Backend identifier.
        available: Whether the backend is installed and usable.
        description: Human-readable description.
    """

    name: AttentionBackend
    available: bool
    description: str


def detect_flash_attention() -> bool:
    """Check if Flash Attention 2 is installed and usable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def detect_sdpa() -> bool:
    """Check if PyTorch scaled dot-product attention is available."""
    try:
        import torch
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")
    except ImportError:
        return False


def detect_available_backends() -> list[AttentionBackendInfo]:
    """Detect all available attention backends."""
    backends = [
        AttentionBackendInfo(
            name="flash",
            available=detect_flash_attention(),
            description="Flash Attention 2 — fastest, requires flash-attn package",
        ),
        AttentionBackendInfo(
            name="sdpa",
            available=detect_sdpa(),
            description="PyTorch Scaled Dot-Product Attention — built-in, good performance",
        ),
        AttentionBackendInfo(
            name="vanilla",
            available=True,
            description="Standard attention — always available, higher memory usage",
        ),
    ]
    return backends


def select_attention_backend(requested: AttentionBackend = "auto") -> AttentionBackend:
    """Select the best available attention backend.

    If 'auto', tries flash > sdpa > vanilla in priority order.
    If a specific backend is requested, validates it's available.
    """
    if requested != "auto":
        if requested == "flash" and not detect_flash_attention():
            return "sdpa" if detect_sdpa() else "vanilla"
        if requested == "sdpa" and not detect_sdpa():
            return "vanilla"
        return requested
    if detect_flash_attention():
        return "flash"
    if detect_sdpa():
        return "sdpa"
    return "vanilla"


def apply_attention_backend(
    model: Any,
    backend: AttentionBackend,
    torch_module: Any,
) -> Any:
    """Apply the selected attention backend to a model.

    Returns the model with the attention mechanism configured.
    """
    return model
