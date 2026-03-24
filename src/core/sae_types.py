"""Sparse Autoencoder options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SaeTrainOptions:
    model_path: str
    output_dir: str
    dataset_name: str
    base_model: str | None = None
    layer_index: int = -1
    latent_dim: int = 0  # 0 = auto (4x hidden_dim)
    max_samples: int = 500
    epochs: int = 10
    learning_rate: float = 1e-3
    sparsity_coeff: float = 1e-3


@dataclass(frozen=True)
class SaeAnalyzeOptions:
    model_path: str
    output_dir: str
    sae_path: str
    input_text: str
    base_model: str | None = None
    dataset_name: str = ""
    top_k_features: int = 10
    top_k_texts: int = 3
