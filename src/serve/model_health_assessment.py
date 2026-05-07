"""Advisory assessment logic for model health check outputs."""

from __future__ import annotations

from typing import Literal

from serve.model_health_suite import HealthCheckId

Severity = Literal["ok", "review", "warning", "critical"]


def assess_health_result(check_id: HealthCheckId, result: dict[str, object]) -> dict[str, str]:
    """Turn diagnostic output into consultant-style assessment fields."""
    if check_id == "weight-norms":
        return _assess_weight_norms(result)
    if check_id == "activation-norms":
        return _assess_activation_norms(result)
    if check_id == "gradient-norms":
        return _assess_gradient_norms(result)
    if check_id == "logit-lens":
        return _assess_logit_lens(result)
    if check_id == "activation-pca":
        return _assess_activation_pca(result)
    if check_id == "activation-patching":
        return _assess_activation_patching(result)
    return _assess_linear_probe(result)


def _assess_weight_norms(result: dict[str, object]) -> dict[str, str]:
    flagged = _int_value(result, "flagged_layer_count")
    max_ratio = _float_value(result, "max_layer_norm_ratio")
    if flagged > 0:
        flagged_layers = _dict_list(result.get("flagged_layers"))
        if _has_nonfinite_values(flagged_layers):
            return _assessment(
                f"{flagged} layer groups contain non-finite weights or invalid parameter values.",
                "Non-finite weights are a hard model-integrity issue and can produce undefined behavior.",
                "Block promotion, reload the checkpoint from a known-good artifact, and rerun the scan.",
                "critical",
            )
        return _assessment(
            f"Large finite weight outlier detected in {_format_flagged_layers(flagged_layers)}.",
            "This is a baseline-comparison flag, not evidence of corruption by itself. Pretrained instruct checkpoints can contain large finite tensors.",
            "Compare the flagged tensors against the base model or last stable checkpoint; investigate only if the outlier is new or paired with eval/activation regressions.",
            "review",
        )
    return _assessment(
        f"Weight scan found no layer group above the norm threshold; max ratio was {max_ratio:.2f}x.",
        "No obvious static parameter explosion was detected in the scanned layers.",
        "Keep this scan as a baseline and rerun after additional fine-tuning or merging.",
        "ok",
    )


def _assess_activation_norms(result: dict[str, object]) -> dict[str, str]:
    flagged = _int_value(result, "flagged_layer_count")
    if flagged > 0:
        return _assessment(
            f"{flagged} layers show activation norm spikes on the calibration set.",
            "The model may be amplifying certain prompts internally, which can create brittle or erratic behavior.",
            "Review the flagged layers and rerun with product-specific prompts before promoting this model.",
            "review",
        )
    return _assessment(
        "Activation norms stayed within the expected range for the sampled calibration prompts.",
        "No sampled layer showed obvious saturation or activation explosion.",
        "Use the same calibration set as a baseline when comparing future candidate models.",
        "ok",
    )


def _assess_gradient_norms(result: dict[str, object]) -> dict[str, str]:
    flagged = _int_value(result, "flagged_layer_count")
    if flagged > 0:
        flagged_layers = _dict_list(result.get("flagged_layers"))
        if _has_nonfinite_values(flagged_layers):
            return _assessment(
                f"{flagged} layer groups produced non-finite gradients on the calibration objective.",
                "Non-finite gradients indicate an unstable objective or broken numerical path.",
                "Block further fine-tuning until the prompt batch, precision, and optimizer settings are fixed.",
                "critical",
            )
        return _assessment(
            f"High gradient sensitivity detected in {_format_flagged_layers(flagged_layers)}.",
            "This identifies sensitive layers for the sampled objective; it is not a promotion blocker for an off-the-shelf model unless it repeats on production-like data or after fine-tuning.",
            "If fine-tuning or merging adapters, compare against the base model and use a lower learning rate or stronger gradient clipping only if this sensitivity is new.",
            "review",
        )
    return _assessment(
        "Gradient norms did not show an obvious exploding layer group on the calibration objective.",
        "The sampled objective does not indicate immediate training-instability risk.",
        "Continue monitoring gradient norms during any follow-up fine-tune.",
        "ok",
    )


def _assess_logit_lens(result: dict[str, object]) -> dict[str, str]:
    tokens = len(_list_value(result, "input_tokens"))
    layers = len(_list_value(result, "layers"))
    warning = result.get("warning")
    if isinstance(warning, str) and warning:
        return _assessment(
            "Prediction trace is not release-grade because the probe produced unknown tokens.",
            "The prompt/tokenizer pairing may not reflect real product traffic, so this check is weak evidence.",
            "Replace the probe with text sampled from the calibration dataset and rerun before relying on this result.",
            "warning",
        )
    return _assessment(
        f"Prediction path completed across {layers} layers for {tokens} prompt tokens.",
        "No prediction-trace runner issue was detected, but the result still needs review against expected behavior.",
        "Compare top predictions against the intended task response and keep this prompt as a regression slice.",
        "ok",
    )


def _assess_activation_pca(result: dict[str, object]) -> dict[str, str]:
    variance = sum(_float_values(_list_value(result, "explained_variance")))
    if variance and variance < 0.35:
        return _assessment(
            f"Representation map is diffuse; two PCA components explain {variance * 100:.1f}% of variance.",
            "This is normal for high-dimensional LLM activations and is not a health failure by itself.",
            "Use labels or source fields to check whether clusters align with unwanted metadata or product task slices.",
            "ok",
        )
    points = len(_list_value(result, "points"))
    return _assessment(
        f"Mapped {points} calibration activations for representation review.",
        "The representation check completed, so cluster shape can be reviewed for drift or shortcuts.",
        "Review clusters by label and source; rebalance data if separation follows artifact metadata.",
        "ok",
    )


def _assess_activation_patching(result: dict[str, object]) -> dict[str, str]:
    layers = _dict_list(result.get("layer_results"))
    if not layers:
        return _assessment(
            "Causal contrast returned no layer-level recovery rows.",
            "The runner did not identify where the behavior difference is represented.",
            "Check the clean/corrupted prompts and rerun with a clearer behavioral contrast.",
            "warning",
        )
    best = max(layers, key=lambda item: _float_or_zero(item.get("recovery")))
    layer = best.get("layer_index", "unknown")
    recovery = _float_or_zero(best.get("recovery")) * 100
    if recovery >= 80:
        return _assessment(
            f"Causal contrast is localized; layer {layer} restores {recovery:.1f}% of the contrast.",
            "A localized causal pathway is expected for some behaviors and is not a defect by default.",
            "Keep this prompt pair as a regression slice; compare with the base model if the behavior is product-critical or unwanted.",
            "ok",
        )
    return _assessment(
        f"Tested {len(layers)} layers; strongest causal recovery was {recovery:.1f}% at layer {layer}.",
        "The behavior is not dominated by one obvious layer in this contrast.",
        "Keep the contrast in the release eval set and rerun patching if product behavior changes.",
        "ok",
    )


def _assess_linear_probe(result: dict[str, object]) -> dict[str, str]:
    layers = _dict_list(result.get("layers"))
    if not layers:
        return _assessment(
            "Label separability produced no probe-layer results.",
            "The supervised signal could not be assessed from this run.",
            "Confirm the label field exists with at least two classes, then rerun the supervised suite.",
            "warning",
        )
    best = max(layers, key=lambda item: _float_or_zero(item.get("accuracy")))
    layer = best.get("layer_index", "unknown")
    accuracy = _float_or_zero(best.get("accuracy")) * 100
    if accuracy < 60:
        return _assessment(
            f"Label signal is weak; best probe accuracy was {accuracy:.1f}% at layer {layer}.",
            "The model may not reliably encode the target distinction used by this product workflow.",
            "Add targeted examples or fine-tune/evaluate on this label before promotion.",
            "review",
        )
    return _assessment(
        f"Label signal is present; best probe accuracy was {accuracy:.1f}% at layer {layer}.",
        "The model internally represents the supervised distinction, which supports targeted behavior review.",
        "Validate this label on held-out evals and check that the signal is task-relevant rather than spurious.",
        "ok",
    )


def _assessment(summary: str, implication: str, action: str, severity: Severity) -> dict[str, str]:
    return {
        "summary": summary,
        "implication": implication,
        "recommended_action": action,
        "severity": severity,
    }


def _int_value(result: dict[str, object], key: str) -> int:
    value = result.get(key)
    return int(value) if isinstance(value, int | float) else 0


def _float_value(result: dict[str, object], key: str) -> float:
    value = result.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _list_value(result: dict[str, object], key: str) -> list[object]:
    value = result.get(key)
    return value if isinstance(value, list) else []


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [{str(key): item[key] for key in item} for item in value if isinstance(item, dict)]


def _has_nonfinite_values(layers: list[dict[str, object]]) -> bool:
    return any(_float_or_zero(layer.get("nonfinite_count")) > 0 for layer in layers)


def _format_flagged_layers(layers: list[dict[str, object]]) -> str:
    if not layers:
        return "the scanned layers"
    parts = [_format_flagged_layer(layer) for layer in layers[:3]]
    if len(layers) > 3:
        parts.append(f"{len(layers) - 3} more")
    return ", ".join(parts)


def _format_flagged_layer(layer: dict[str, object]) -> str:
    name = _primary_tensor_name(layer)
    ratio = _float_or_zero(layer.get("norm_ratio"))
    if ratio <= 0:
        return name
    return f"{name} ({ratio:.1f}x median)"


def _primary_tensor_name(layer: dict[str, object]) -> str:
    top_parameters = _dict_list(layer.get("top_parameters"))
    if top_parameters:
        name = top_parameters[0].get("name")
        if isinstance(name, str) and name:
            return name
    return str(layer.get("layer_name") or f"layer {layer.get('layer_index', 'unknown')}")


def _float_values(values: list[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, int | float)]


def _float_or_zero(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
