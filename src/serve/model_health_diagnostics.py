"""Additional model health diagnostics."""

from __future__ import annotations

import math
from typing import Any

from serve.activation_extractor import discover_transformer_layers
from serve.interp_activation_collector import collect_activations
from serve.interp_data_utils import extract_texts
from serve.layer_selection import resolve_layer_selection


def run_weight_norm_scan(
    model_path: str,
    base_model: str | None,
    layer_indices: str,
) -> dict[str, object]:
    """Scan model parameter norms by layer group."""
    model = _load_model(model_path, base_model)
    layer_items = _selected_layers(model, layer_indices)
    layer_stats = [_parameter_stats(model, name, index) for index, name in layer_items]
    norms = [_as_float(stat["weight_norm"]) for stat in layer_stats if _as_int(stat["parameter_count"])]
    median_norm = _median(norms)
    flagged = []
    for stat in layer_stats:
        ratio = _as_float(stat["weight_norm"]) / median_norm if median_norm > 0 else 0.0
        stat["norm_ratio"] = round(ratio, 4)
        if ratio > 3.0 or _as_int(stat["nonfinite_count"]) > 0:
            flagged.append(stat)
    return {
        "check_type": "weight-norms",
        "layers": layer_stats,
        "median_layer_norm": round(median_norm, 6),
        "max_layer_norm_ratio": round(max([_as_float(s.get("norm_ratio", 0.0)) for s in layer_stats], default=0.0), 4),
        "flagged_layer_count": len(flagged),
        "flagged_layers": flagged,
    }


def run_activation_norm_scan(
    model_path: str,
    base_model: str | None,
    records: list[object],
    max_samples: int,
    layer_indices: str,
) -> dict[str, object]:
    """Scan activation norm distributions across selected layers."""
    model, tokenizer = _load_model_and_tokenizer(model_path, base_model)
    layer_items = _selected_layers(model, layer_indices)
    layer_stats = []
    for layer_index, layer_name in layer_items:
        stacked, _, _ = collect_activations(
            model, tokenizer, records, layer_name, min(max_samples, 64), granularity="sample",
        )
        norms = stacked.float().norm(dim=1).tolist()
        layer_stats.append(_norm_summary(layer_index, layer_name, norms))
    median_mean = _median([_as_float(stat.get("mean_norm", 0.0)) for stat in layer_stats])
    flagged = [
        stat for stat in layer_stats
        if median_mean > 0 and _as_float(stat.get("mean_norm", 0.0)) / median_mean > 3.0
    ]
    return {
        "check_type": "activation-norms",
        "layers": layer_stats,
        "median_mean_norm": round(median_mean, 6),
        "flagged_layer_count": len(flagged),
        "flagged_layers": flagged,
    }


def run_gradient_norm_scan(
    model_path: str,
    base_model: str | None,
    records: list[object],
    max_samples: int,
    layer_indices: str,
) -> dict[str, object]:
    """Backprop a next-token objective and scan layer gradient norms."""
    import torch.nn.functional as functional

    model, tokenizer = _load_model_and_tokenizer(model_path, base_model)
    layer_items = _selected_layers(model, layer_indices)
    texts = extract_texts(records, min(max_samples, 8))
    if not texts:
        raise ValueError("No usable text found for gradient norm scan.")
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    device = next(model.parameters()).device
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        if input_ids.shape[1] < 2:
            continue
        logits = _extract_logits(model(input_ids))
        loss = functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss_any: Any = loss
        loss_any.backward()
        losses.append(float(loss.detach().cpu().item()))
    layer_stats = [_gradient_stats(model, name, index) for index, name in layer_items]
    norms = [_as_float(stat["gradient_norm"]) for stat in layer_stats if _as_int(stat["parameter_count"])]
    median_norm = _median(norms)
    flagged = []
    for stat in layer_stats:
        ratio = _as_float(stat["gradient_norm"]) / median_norm if median_norm > 0 else 0.0
        stat["norm_ratio"] = round(ratio, 4)
        if ratio > 5.0 or _as_int(stat["nonfinite_count"]) > 0:
            flagged.append(stat)
    model.eval()
    return {
        "check_type": "gradient-norms",
        "loss_mean": round(sum(losses) / max(len(losses), 1), 6),
        "layers": layer_stats,
        "median_layer_gradient_norm": round(median_norm, 6),
        "flagged_layer_count": len(flagged),
        "flagged_layers": flagged,
    }


def _load_model(model_path: str, base_model: str | None) -> Any:
    from serve.interp_model_loader import load_interp_model

    model, _ = load_interp_model(base_model or model_path)
    model.eval()
    return model


def _load_model_and_tokenizer(model_path: str, base_model: str | None) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model

    model, tokenizer = load_interp_model(base_model or model_path)
    model.eval()
    return model, tokenizer


def _selected_layers(model: Any, layer_indices: str) -> list[tuple[int, str]]:
    return resolve_layer_selection(discover_transformer_layers(model), layer_indices)


def _parameter_stats(model: Any, layer_name: str, layer_index: int) -> dict[str, object]:
    params = _matching_parameters(model, layer_name)
    total_sq = 0.0
    max_abs = 0.0
    nonfinite = 0
    count = 0
    parameter_rows: list[dict[str, object]] = []
    for name, parameter in params:
        data = parameter.detach().float()
        param_sq = float((data * data).sum().cpu().item())
        param_max = float(data.abs().max().cpu().item())
        param_nonfinite = int((~data.isfinite()).sum().cpu().item())
        total_sq += param_sq
        max_abs = max(max_abs, param_max)
        nonfinite += param_nonfinite
        count += data.numel()
        parameter_rows.append({
            "name": name,
            "parameter_count": data.numel(),
            "norm": round(math.sqrt(param_sq), 6),
            "max_abs": round(param_max, 6),
            "nonfinite_count": param_nonfinite,
        })
    return {
        "layer_index": layer_index,
        "layer_name": layer_name,
        "parameter_count": count,
        "weight_norm": round(math.sqrt(total_sq), 6),
        "max_abs_weight": round(max_abs, 6),
        "nonfinite_count": nonfinite,
        "top_parameters": _top_norm_rows(parameter_rows),
    }


def _gradient_stats(model: Any, layer_name: str, layer_index: int) -> dict[str, object]:
    total_sq = 0.0
    nonfinite = 0
    count = 0
    parameter_rows: list[dict[str, object]] = []
    for name, parameter in _matching_parameters(model, layer_name):
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        param_sq = float((grad * grad).sum().cpu().item())
        param_nonfinite = int((~grad.isfinite()).sum().cpu().item())
        total_sq += param_sq
        nonfinite += param_nonfinite
        count += grad.numel()
        parameter_rows.append({
            "name": name,
            "parameter_count": grad.numel(),
            "norm": round(math.sqrt(param_sq), 6),
            "nonfinite_count": param_nonfinite,
        })
    return {
        "layer_index": layer_index,
        "layer_name": layer_name,
        "parameter_count": count,
        "gradient_norm": round(math.sqrt(total_sq), 6),
        "nonfinite_count": nonfinite,
        "top_parameters": _top_norm_rows(parameter_rows),
    }


def _matching_parameters(model: Any, layer_name: str) -> list[tuple[str, Any]]:
    prefix = f"{layer_name}."
    return [(name, param) for name, param in model.named_parameters() if name.startswith(prefix)]


def _norm_summary(layer_index: int, layer_name: str, norms: list[float]) -> dict[str, object]:
    if not norms:
        return {"layer_index": layer_index, "layer_name": layer_name, "sample_count": 0}
    ordered = sorted(float(norm) for norm in norms)
    return {
        "layer_index": layer_index,
        "layer_name": layer_name,
        "sample_count": len(ordered),
        "mean_norm": round(sum(ordered) / len(ordered), 6),
        "max_norm": round(max(ordered), 6),
        "p95_norm": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 6),
    }


def _top_norm_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: _as_float(row.get("norm")), reverse=True)[:3]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0
