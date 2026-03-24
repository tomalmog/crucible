"""SAE analysis runner: decompose a text's activations into SAE features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from core.sae_types import SaeAnalyzeOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers
from serve.interp_data_utils import extract_single_text
from serve.sae_model import load_sae


def run_sae_analyze(
    options: SaeAnalyzeOptions, records: list[Any] | None = None,
) -> dict[str, Any]:
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

    # Encode the user's input text
    pooled = _encode_text(model, tokenizer, options.input_text, target_layer, device)

    with torch.no_grad():
        reconstruction, latent = sae(pooled)

    recon_error = float(torch.nn.functional.mse_loss(reconstruction, pooled))
    sparsity = float((latent > 0).float().mean())
    active_features = int((latent[0] > 0).sum())

    # Top-K features by activation strength
    values, indices = latent[0].topk(min(options.top_k_features, latent.shape[1]))
    top_indices = [int(idx) for idx, val in zip(indices.tolist(), values.tolist()) if float(val) > 0]
    features = [
        {"feature_index": int(idx), "activation": round(float(val), 6)}
        for idx, val in zip(indices.tolist(), values.tolist())
        if float(val) > 0
    ]

    # Build feature→text associations from training dataset
    if records and top_indices:
        associations = _build_feature_associations(
            model, tokenizer, sae, records, target_layer, device,
            top_indices, options.top_k_texts,
        )
        for feat in features:
            feat["associated_texts"] = associations.get(feat["feature_index"], [])

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


def _encode_text(
    model: Any, tokenizer: Any, text: str,
    target_layer: str, device: Any,
) -> torch.Tensor:
    """Encode text through the model and return mean-pooled activation."""
    with torch.no_grad():
        with ActivationExtractor(model, [target_layer]) as extractor:
            ids = tokenizer.encode(
                text, return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)
            model(ids)
            acts = extractor.get_activations()[target_layer]
    return acts[0].mean(dim=0).unsqueeze(0).to(dtype=torch.float32)


def _build_feature_associations(
    model: Any, tokenizer: Any, sae: Any,
    records: list[Any], target_layer: str, device: Any,
    feature_indices: list[int], top_k_texts: int,
) -> dict[int, list[str]]:
    """For each feature index, find the training texts that activate it most."""
    # Encode all records through model + SAE
    texts: list[str] = []
    latents: list[torch.Tensor] = []

    for record in records:
        text = extract_single_text(record)
        if not text:
            continue
        texts.append(text)
        pooled = _encode_text(model, tokenizer, text, target_layer, device)
        with torch.no_grad():
            _, lat = sae(pooled)
        latents.append(lat[0])

    if not latents:
        return {}

    # Stack into (num_texts, latent_dim)
    all_latents = torch.stack(latents)

    # For each feature, find top-k texts by activation
    associations: dict[int, list[str]] = {}
    for feat_idx in feature_indices:
        col = all_latents[:, feat_idx]
        k = min(top_k_texts, len(texts))
        top_vals, top_text_indices = col.topk(k)
        associated = []
        for ti, tv in zip(top_text_indices.tolist(), top_vals.tolist()):
            if tv > 0:
                snippet = texts[ti][:120]
                associated.append(snippet)
        associations[feat_idx] = associated

    return associations


def _load_model_and_tokenizer(options: SaeAnalyzeOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
