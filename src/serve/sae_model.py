"""Sparse Autoencoder architecture and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder with ReLU latent activations."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent = torch.relu(self.encoder(x))
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def save_sae(sae: SparseAutoencoder, path: Path, metadata: dict[str, Any]) -> None:
    """Save SAE weights and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))


def load_sae(path: Path, device: str = "cpu") -> tuple[SparseAutoencoder, dict[str, Any]]:
    """Load SAE weights and metadata."""
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    sae = SparseAutoencoder(metadata["input_dim"], metadata["latent_dim"])
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    sae.to(device)
    return sae, metadata
