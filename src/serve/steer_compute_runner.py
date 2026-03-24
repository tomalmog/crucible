"""Steering vector computation: derive direction from contrastive examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from core.steering_types import SteerComputeOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers
from serve.interp_data_utils import extract_texts
from serve.steering_vector_io import save_steering_vector


def run_steer_compute(
    options: SteerComputeOptions,
    positive_records: list[Any] | None = None,
    negative_records: list[Any] | None = None,
) -> dict[str, Any]:
    """Compute a steering vector from contrastive examples."""
    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    layer_idx = options.layer_index if options.layer_index >= 0 else len(all_layers) - 1
    target_layer = all_layers[layer_idx]

    # Gather positive and negative texts
    if positive_records and negative_records:
        pos_texts = extract_texts(positive_records, options.max_samples)
        neg_texts = extract_texts(negative_records, options.max_samples)
    else:
        pos_texts = [options.positive_text] if options.positive_text else []
        neg_texts = [options.negative_text] if options.negative_text else []

    if not pos_texts or not neg_texts:
        from core.errors import CrucibleError
        raise CrucibleError("Need both positive and negative text for steering vector computation.")

    pos_mean = _mean_activation(model, tokenizer, pos_texts, target_layer)
    neg_mean = _mean_activation(model, tokenizer, neg_texts, target_layer)
    steering_vector = pos_mean - neg_mean

    vector_norm = float(steering_vector.norm())
    cosine_sim = float(torch.nn.functional.cosine_similarity(
        pos_mean.unsqueeze(0), neg_mean.unsqueeze(0),
    ))

    out_dir = Path(options.output_dir).expanduser().resolve()
    vector_path = out_dir / "steering_vector.pt"
    metadata = {
        "layer_name": target_layer,
        "layer_index": layer_idx,
        "num_positive": len(pos_texts),
        "num_negative": len(neg_texts),
        "vector_norm": round(vector_norm, 6),
    }
    save_steering_vector(steering_vector, vector_path, metadata)

    result: dict[str, Any] = {
        "steering_vector_path": str(vector_path),
        "layer_name": target_layer,
        "layer_index": layer_idx,
        "vector_norm": round(vector_norm, 6),
        "cosine_similarity": round(cosine_sim, 6),
        "num_positive": len(pos_texts),
        "num_negative": len(neg_texts),
    }

    out_path = out_dir / "steer_compute.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _mean_activation(
    model: Any, tokenizer: Any, texts: list[str], target_layer: str,
) -> Any:
    """Compute mean activation across texts at a layer."""
    device = next(model.parameters()).device
    vectors = []

    with torch.no_grad():
        with ActivationExtractor(model, [target_layer]) as extractor:
            for text in texts:
                extractor.clear()
                ids = tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=512,
                )
                ids = ids.to(device)
                model(ids)
                acts = extractor.get_activations()[target_layer]
                pooled = acts[0].mean(dim=0).cpu()
                vectors.append(pooled)

    return torch.stack(vectors).mean(dim=0)


def _load_model_and_tokenizer(options: SteerComputeOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
