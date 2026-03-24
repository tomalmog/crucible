"""Steering vector save/load utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def save_steering_vector(
    vector: Tensor, path: Path, metadata: dict[str, Any],
) -> None:
    """Save a steering vector and its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))


def load_steering_vector(
    path: Path, device: str = "cpu",
) -> tuple[Tensor, dict[str, Any]]:
    """Load a steering vector and its metadata."""
    vector = torch.load(path, map_location=device, weights_only=True)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    return vector, metadata
