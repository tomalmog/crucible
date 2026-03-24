"""Activation PCA runner: extract activations, project to 2D via PCA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.activation_pca_types import ActivationPcaOptions
from serve.activation_extractor import discover_transformer_layers
from serve.interp_activation_collector import collect_activations


def run_activation_pca(
    options: ActivationPcaOptions, records: list[Any],
) -> dict[str, Any]:
    """Run PCA on layer activations and write results JSON."""
    import torch

    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    layer_idx = options.layer_index if options.layer_index >= 0 else len(all_layers) - 1
    target_layer = all_layers[layer_idx]

    stacked, labels, snippets = collect_activations(
        model, tokenizer, records, target_layer, options.max_samples,
        granularity=options.granularity, label_field=options.color_field,
    )

    n_components = min(2, stacked.shape[0], stacked.shape[1])

    # PCA via SVD — no sklearn dependency needed
    mean = stacked.mean(dim=0)
    centered = stacked - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    projected = (centered @ Vh[:n_components].T).numpy()
    total_var = (S ** 2).sum()
    explained_variance = [(float(S[i] ** 2 / total_var)) for i in range(n_components)]

    points = []
    for i in range(len(projected)):
        points.append({
            "x": round(float(projected[i, 0]), 4),
            "y": round(float(projected[i, 1]), 4) if n_components > 1 else 0.0,
            "label": labels[i],
            "text": snippets[i],
        })

    result: dict[str, Any] = {
        "layer_name": target_layer,
        "layer_index": layer_idx,
        "granularity": options.granularity,
        "explained_variance": [round(v, 4) for v in explained_variance],
        "points": points,
    }

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "activation_pca.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _load_model_and_tokenizer(options: ActivationPcaOptions) -> tuple[Any, Any]:
    """Load model + tokenizer for PCA analysis."""
    from serve.interp_model_loader import load_interp_model

    model_path = options.base_model or options.model_path
    return load_interp_model(model_path)
