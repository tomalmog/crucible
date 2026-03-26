"""Logit lens runner: project each layer's hidden states through unembedding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.logit_lens_types import LogitLensOptions
from serve.activation_extractor import (
    ActivationExtractor,
    discover_transformer_layers,
    get_unembedding,
)


def run_logit_lens(options: LogitLensOptions) -> dict[str, Any]:
    """Run logit lens analysis and write results JSON."""
    import torch

    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    target_layers = _resolve_layer_indices(all_layers, options.layer_indices)

    input_ids = tokenizer.encode(options.input_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    unembed = get_unembedding(model)

    with torch.no_grad():
        with ActivationExtractor(model, target_layers) as extractor:
            model(input_ids)
            activations = extractor.get_activations()

    layers_result = []
    for layer_name in target_layers:
        hidden = activations[layer_name]  # [1, seq_len, hidden_dim]
        unembed_device = next(unembed.parameters()).device
        logits = unembed(hidden.to(unembed_device))  # [1, seq_len, vocab]
        probs = torch.softmax(logits[0], dim=-1)
        layer_idx = all_layers.index(layer_name)

        predictions = []
        for pos in range(probs.shape[0]):
            top_probs, top_ids = torch.topk(probs[pos], options.top_k)
            top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
            predictions.append({
                "token_position": pos,
                "top_k": [
                    {"token": tok, "prob": round(p.item(), 4)}
                    for tok, p in zip(top_tokens, top_probs)
                ],
            })
        layers_result.append({
            "layer_index": layer_idx,
            "layer_name": layer_name,
            "predictions": predictions,
        })

    result: dict[str, Any] = {"input_tokens": input_tokens, "layers": layers_result}

    # Warn if many input tokens are unknown
    unk_count = sum(1 for t in input_tokens if t in ("<unk>", "[UNK]", "<|endoftext|>"))
    if unk_count > 0:
        result["warning"] = (
            f"{unk_count} of {len(input_tokens)} input tokens are unknown (<unk>). "
            "This model's vocabulary may not cover the input text. "
            "Try using words from the model's training data for meaningful results."
        )

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logit_lens.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _load_model_and_tokenizer(options: LogitLensOptions) -> tuple[Any, Any]:
    """Load model + tokenizer for interpretability analysis."""
    from serve.interp_model_loader import load_interp_model

    model_path = options.base_model or options.model_path
    return load_interp_model(model_path)


def _resolve_layer_indices(
    all_layers: list[str], indices_str: str,
) -> list[str]:
    """Parse comma-separated layer indices into layer names."""
    from core.errors import CrucibleServeError

    if not indices_str.strip():
        return all_layers
    indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
    resolved = [all_layers[i] for i in indices if 0 <= i < len(all_layers)]
    if not resolved:
        raise CrucibleServeError(
            f"No valid layer indices found. Model has {len(all_layers)} layers, "
            f"requested: {indices_str}"
        )
    return resolved
