"""SAE analysis runner: decompose a text's activations into SAE features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from core.sae_types import SaeAnalyzeOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers
from serve.sae_model import load_sae


def run_sae_analyze(options: SaeAnalyzeOptions) -> dict[str, Any]:
    """Analyze input text through a trained SAE."""
    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    sae_path = Path(options.sae_path).expanduser().resolve()
    device_str = str(next(model.parameters()).device)
    sae, sae_meta = load_sae(sae_path, device=device_str)
    sae.eval()

    target_layer = sae_meta["layer_name"]
    all_layers = discover_transformer_layers(model)
    if target_layer not in all_layers:
        layer_idx = sae_meta.get("layer_index", len(all_layers) - 1)
        target_layer = all_layers[layer_idx]

    device = next(model.parameters()).device
    with torch.no_grad():
        with ActivationExtractor(model, [target_layer]) as extractor:
            ids = tokenizer.encode(
                options.input_text, return_tensors="pt",
                truncation=True, max_length=512,
            )
            ids = ids.to(device)
            model(ids)
            acts = extractor.get_activations()[target_layer]

    # Mean-pool across sequence
    pooled = acts[0].mean(dim=0).unsqueeze(0)

    with torch.no_grad():
        reconstruction, latent = sae(pooled)

    recon_error = float(torch.nn.functional.mse_loss(reconstruction, pooled))
    sparsity = float((latent > 0).float().mean())
    active_features = int((latent[0] > 0).sum())

    # Top-K features by activation strength
    values, indices = latent[0].topk(min(options.top_k_features, latent.shape[1]))
    features = [
        {"feature_index": int(idx), "activation": round(float(val), 6)}
        for idx, val in zip(indices.tolist(), values.tolist())
        if float(val) > 0
    ]

    result: dict[str, Any] = {
        "input_text": options.input_text[:200],
        "layer_name": target_layer,
        "reconstruction_error": round(recon_error, 6),
        "sparsity": round(sparsity, 4),
        "active_features": active_features,
        "total_features": sae_meta["latent_dim"],
        "top_features": features,
    }

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sae_analyze.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _load_model_and_tokenizer(options: SaeAnalyzeOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
