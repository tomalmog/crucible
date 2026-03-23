"""Activation PCA runner: extract activations, project to 2D via PCA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.activation_pca_types import ActivationPcaOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers


def run_activation_pca(
    options: ActivationPcaOptions, records: list[Any],
) -> dict[str, Any]:
    """Run PCA on layer activations and write results JSON."""
    import torch
    from sklearn.decomposition import PCA  # type: ignore[import-untyped,unused-ignore]

    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    layer_idx = options.layer_index if options.layer_index >= 0 else len(all_layers) - 1
    target_layer = all_layers[layer_idx]
    device = next(model.parameters()).device

    texts = _extract_texts(records, options.max_samples)
    vectors: list[Any] = []
    labels: list[str] = []
    snippets: list[str] = []

    with torch.no_grad():
        with ActivationExtractor(model, [target_layer]) as extractor:
            for i, text in enumerate(texts):
                extractor.clear()
                ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
                ids = ids.to(device)
                model(ids)
                acts = extractor.get_activations()[target_layer]  # [1, seq, hidden]

                color = _get_color_label(records, i, options.color_field)

                if options.granularity == "token":
                    token_strs = tokenizer.convert_ids_to_tokens(ids[0].tolist())
                    for t in range(acts.shape[1]):
                        vectors.append(acts[0, t].cpu())
                        labels.append(color)
                        snippets.append(token_strs[t] if t < len(token_strs) else "")
                else:
                    pooled = acts[0].mean(dim=0).cpu()
                    vectors.append(pooled)
                    labels.append(color)
                    snippets.append(text[:80])

    stacked = torch.stack(vectors).numpy()
    n_components = min(2, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(stacked)

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
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
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


def _extract_texts(records: list[Any], max_samples: int) -> list[str]:
    """Extract text content from dataset records."""
    texts: list[str] = []
    for record in records[:max_samples]:
        # Crucible DataRecord has .text; remote _SimpleRecord has .content
        text = getattr(record, "text", None) or getattr(record, "content", None)
        if isinstance(text, str) and text.strip():
            texts.append(text)
        elif isinstance(text, dict):
            val = text.get("text", "") or text.get("input", "")
            if val:
                texts.append(str(val))
    return texts


def _get_color_label(records: list[Any], index: int, field: str) -> str:
    """Extract a color label from record metadata."""
    if not field or index >= len(records):
        return ""
    record = records[index]
    meta = getattr(record, "metadata", None) or {}
    if isinstance(meta, dict):
        return str(meta.get(field, ""))
    return ""
