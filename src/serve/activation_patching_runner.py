"""Activation patching runner: find causally important layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.activation_patching_types import ActivationPatchingOptions
from serve.activation_extractor import ActivationExtractor, discover_transformer_layers


def run_activation_patching(options: ActivationPatchingOptions) -> dict[str, Any]:
    """Run activation patching and write results JSON."""
    import torch

    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    all_layers = discover_transformer_layers(model)
    device = next(model.parameters()).device

    clean_ids = tokenizer.encode(options.clean_text, return_tensors="pt").to(device)
    corrupt_ids = tokenizer.encode(options.corrupted_text, return_tensors="pt").to(device)

    target_idx = options.target_token_index

    with torch.no_grad():
        # Clean run: collect all layer activations + final logits
        with ActivationExtractor(model, all_layers) as extractor:
            clean_out = model(clean_ids)
            clean_acts = extractor.get_activations()

        clean_logits = _extract_logits(clean_out, target_idx)

        # Corrupted run: final logits only
        corrupt_out = model(corrupt_ids)
        corrupt_logits = _extract_logits(corrupt_out, target_idx)

    clean_metric = _compute_metric(clean_logits, options.metric)
    corrupt_metric = _compute_metric(corrupt_logits, options.metric)
    metric_range = clean_metric - corrupt_metric

    layer_results = []
    for layer_name in all_layers:
        patched_logits = _run_patched_forward(
            model, corrupt_ids, layer_name,
            clean_acts[layer_name], target_idx,
        )
        patched_metric = _compute_metric(patched_logits, options.metric)
        recovery = 0.0
        if abs(metric_range) > 1e-8:
            recovery = (patched_metric - corrupt_metric) / metric_range

        layer_results.append({
            "layer_index": all_layers.index(layer_name),
            "layer_name": layer_name,
            "patched_metric": round(patched_metric, 4),
            "recovery": round(recovery, 4),
        })

    result: dict[str, Any] = {
        "clean_text": options.clean_text,
        "corrupted_text": options.corrupted_text,
        "metric": options.metric,
        "clean_metric": round(clean_metric, 4),
        "corrupted_metric": round(corrupt_metric, 4),
        "layer_results": layer_results,
    }

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "activation_patching.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _run_patched_forward(
    model: Any, input_ids: Any, target_layer: str,
    clean_activation: Any, target_idx: int,
) -> Any:
    """Forward corrupted input with one layer's activation replaced by clean."""
    import torch

    patched_act = clean_activation

    def patch_hook(_module: Any, _input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            patched = list(output)
            seq_len = min(patched_act.shape[1], patched[0].shape[1])
            patched[0] = patched[0].clone()
            patched[0][:, :seq_len] = patched_act[:, :seq_len]
            return tuple(patched)
        seq_len = min(patched_act.shape[1], output.shape[1])
        result = output.clone()
        result[:, :seq_len] = patched_act[:, :seq_len]
        return result

    # Resolve module and attach temporary hook
    parts = target_layer.split(".")
    module = model
    for part in parts:
        module = module[int(part)] if part.isdigit() else getattr(module, part)

    handle = module.register_forward_hook(patch_hook)
    with torch.no_grad():
        out = model(input_ids)
    handle.remove()

    return _extract_logits(out, target_idx)


def _extract_logits(output: Any, target_idx: int) -> Any:
    """Extract logits at target token position from model output."""
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    return logits[0, target_idx]


def _compute_metric(logits: Any, metric: str) -> float:
    """Compute scalar metric from logits at target position."""
    import torch

    if metric == "prob":
        probs = torch.softmax(logits, dim=-1)
        return float(probs.max().item())
    # logit_diff: difference between top-2 logits
    top2 = torch.topk(logits, 2)
    return float((top2.values[0] - top2.values[1]).item())


def _load_model_and_tokenizer(
    options: ActivationPatchingOptions,
) -> tuple[Any, Any]:
    """Load model + tokenizer for patching analysis."""
    from serve.interp_model_loader import load_interp_model

    model_path = options.base_model or options.model_path
    return load_interp_model(model_path)
