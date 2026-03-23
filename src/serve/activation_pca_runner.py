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

    texts = _extract_texts(records, options.max_samples)
    if not texts:
        from core.errors import CrucibleError
        raise CrucibleError(
            f"No usable text found in {len(records)} dataset records. "
            "Ensure the dataset has 'text', 'prompt', 'content', or 'input' fields."
        )

    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    layer_idx = options.layer_index if options.layer_index >= 0 else len(all_layers) - 1
    target_layer = all_layers[layer_idx]
    device = next(model.parameters()).device
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

    stacked = torch.stack(vectors).float()
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


_TEXT_ATTR_NAMES = ("text", "content", "prompt", "instruction", "input")


def _extract_texts(records: list[Any], max_samples: int) -> list[str]:
    """Extract text content from dataset records of any format.

    Supports: Crucible DataRecord (.text), remote _SimpleRecord (.content),
    SFT/LoRA/QLoRA records (.prompt + .response), plain dicts, etc.
    """
    texts: list[str] = []
    for record in records[:max_samples]:
        text = _extract_single_text(record)
        if text:
            texts.append(text)
    return texts


def _extract_single_text(record: Any) -> str:
    """Extract a single text string from a record of any format."""
    # Try attribute access (DataRecord, _SimpleRecord, etc.)
    for attr in _TEXT_ATTR_NAMES:
        val = getattr(record, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Try dict access
    if isinstance(record, dict):
        for key in _TEXT_ATTR_NAMES:
            val = record.get(key, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Nested dict: {"text": {"text": "..."}}
        for key in ("text", "content"):
            val = record.get(key)
            if isinstance(val, dict):
                inner = val.get("text", "") or val.get("input", "")
                if inner:
                    return str(inner).strip()

    return ""


def _get_color_label(records: list[Any], index: int, field: str) -> str:
    """Extract a color label from record metadata."""
    if not field or index >= len(records):
        return ""
    record = records[index]
    meta = getattr(record, "metadata", None) or {}
    if isinstance(meta, dict):
        return str(meta.get(field, ""))
    return ""
