"""SAE training runner: train a sparse autoencoder on layer activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from core.sae_types import SaeTrainOptions
from serve.activation_extractor import discover_transformer_layers
from serve.interp_activation_collector import collect_activations
from serve.sae_model import SparseAutoencoder, save_sae


def run_sae_train(
    options: SaeTrainOptions, records: list[Any],
) -> dict[str, Any]:
    """Train an SAE on frozen activations and save the model."""
    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    layer_idx = options.layer_index if options.layer_index >= 0 else len(all_layers) - 1
    target_layer = all_layers[layer_idx]

    X, _, _ = collect_activations(
        model, tokenizer, records, target_layer,
        options.max_samples, granularity="sample",
    )
    X = X.to(dtype=torch.float32)

    hidden_dim = X.shape[1]
    latent_dim = options.latent_dim if options.latent_dim > 0 else hidden_dim * 4

    sae = SparseAutoencoder(hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(sae.parameters(), lr=options.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(options.epochs):
        sae.train()
        optimizer.zero_grad()
        reconstruction, latent = sae(X)
        recon_loss = torch.nn.functional.mse_loss(reconstruction, X)
        sparsity_loss = latent.abs().mean()
        loss = recon_loss + options.sparsity_coeff * sparsity_loss
        loss.backward()
        optimizer.step()
        history.append({
            "epoch": epoch + 1,
            "loss": round(float(loss.detach()), 6),
            "recon_loss": round(float(recon_loss.detach()), 6),
            "sparsity_loss": round(float(sparsity_loss.detach()), 6),
        })

    out_dir = Path(options.output_dir).expanduser().resolve()
    sae_path = out_dir / "sae_model.pt"
    metadata = {
        "input_dim": hidden_dim,
        "latent_dim": latent_dim,
        "layer_name": target_layer,
        "layer_index": layer_idx,
        "epochs": options.epochs,
        "sparsity_coeff": options.sparsity_coeff,
    }
    save_sae(sae, sae_path, metadata)

    result: dict[str, Any] = {
        "sae_path": str(sae_path),
        "layer_name": target_layer,
        "layer_index": layer_idx,
        "input_dim": hidden_dim,
        "latent_dim": latent_dim,
        "epochs": options.epochs,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "final_recon_loss": history[-1]["recon_loss"] if history else 0.0,
        "final_sparsity_loss": history[-1]["sparsity_loss"] if history else 0.0,
        "history": history,
    }

    out_path = out_dir / "sae_train.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _load_model_and_tokenizer(options: SaeTrainOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
