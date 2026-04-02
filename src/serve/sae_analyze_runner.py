"""SAE analysis runner: decompose a text's activations into SAE features."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from core.sae_types import SaeAnalyzeOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers
from serve.interp_data_utils import extract_single_text
from serve.sae_model import load_sae

# Maximum number of dataset records to process for feature associations.
# Keeps analysis runtime bounded on large datasets.
_MAX_ASSOCIATION_SAMPLES = 200

# Common stop words to exclude from concept labels
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "as", "if", "not", "no", "so", "up",
    "out", "about", "into", "over", "after", "before", "between", "under",
    "during", "through", "above", "below", "than", "very", "just", "also",
    "then", "more", "most", "such", "only", "other", "new", "one", "two",
    "three", "her", "his", "she", "he", "they", "them", "their", "my",
    "your", "our", "who", "which", "what", "when", "where", "how", "all",
    "each", "every", "both", "few", "many", "some", "any", "while",
})


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

    # Build feature→concept associations from training dataset
    if records and top_indices:
        associations = _build_feature_associations(
            model, tokenizer, sae, records, target_layer, device,
            top_indices, options.top_k_texts,
        )
        for feat in features:
            info = associations.get(feat["feature_index"])
            if info:
                feat["concept"] = info["concept"]
                feat["associated_texts"] = info["texts"]

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
    max_samples: int = _MAX_ASSOCIATION_SAMPLES,
) -> dict[int, dict[str, Any]]:
    """For each feature, find top activating texts and extract a concept label."""
    texts: list[str] = []
    latents: list[torch.Tensor] = []

    capped_records = records[:max_samples]
    for record in capped_records:
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

    all_latents = torch.stack(latents)

    # Build word frequency across ALL texts for TF-IDF-like scoring
    global_word_counts: Counter[str] = Counter()
    for t in texts:
        global_word_counts.update(set(_extract_words(t)))
    num_texts = len(texts)

    associations: dict[int, dict[str, Any]] = {}
    for feat_idx in feature_indices:
        col = all_latents[:, feat_idx]
        k = min(top_k_texts, len(texts))
        top_vals, top_text_indices = col.topk(k)

        top_texts = []
        for ti, tv in zip(top_text_indices.tolist(), top_vals.tolist()):
            if tv > 0:
                top_texts.append(texts[ti])

        concept = _extract_concept(top_texts, global_word_counts, num_texts)
        associations[feat_idx] = {
            "concept": concept,
            "texts": [t[:120] for t in top_texts],
        }

    return associations


def _extract_words(text: str) -> list[str]:
    """Extract lowercase alphabetic words from text."""
    return [w for w in text.lower().split() if w.isalpha() and len(w) > 2]


def _extract_concept(
    top_texts: list[str], global_counts: Counter[str], num_texts: int,
) -> str:
    """Extract a short concept label from the top activating texts.

    Uses TF-IDF-like scoring: words that appear in the top texts but
    are rare across the full dataset are most distinctive.
    """
    if not top_texts:
        return "unknown"

    # Count word frequency in top texts
    local_counts: Counter[str] = Counter()
    for t in top_texts:
        local_counts.update(_extract_words(t))

    # Score each word: local_freq * inverse_global_freq
    scored: list[tuple[str, float]] = []
    for word, local_freq in local_counts.items():
        if word in _STOP_WORDS:
            continue
        global_freq = global_counts.get(word, 1)
        # TF-IDF inspired: how distinctive is this word?
        score = local_freq * (num_texts / global_freq)
        scored.append((word, score))

    scored.sort(key=lambda x: -x[1])

    # Take top 2-3 distinctive words as the concept
    top_words = [w for w, _ in scored[:3]]
    if not top_words:
        return "unknown"
    return " / ".join(top_words)


def _load_model_and_tokenizer(options: SaeAnalyzeOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
