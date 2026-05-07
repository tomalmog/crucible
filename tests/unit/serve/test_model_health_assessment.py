"""Tests for model health advisory wording."""

from __future__ import annotations

from serve.model_health_assessment import assess_health_result


def test_weight_outlier_is_baseline_review() -> None:
    assessment = assess_health_result("weight-norms", _flagged_result("weight_norm"))

    assert (assessment["severity"], "self_attn.k_proj.bias" in assessment["summary"]) == ("review", True)


def test_gradient_outlier_is_not_generic_lr_instruction() -> None:
    assessment = assess_health_result("gradient-norms", _flagged_result("gradient_norm"))

    assert "off-the-shelf model" in assessment["implication"]


def test_diffuse_pca_is_informational() -> None:
    assessment = assess_health_result("activation-pca", {"explained_variance": [0.2, 0.046]})

    assert assessment["severity"] == "ok"


def test_localized_causal_contrast_is_informational() -> None:
    assessment = assess_health_result(
        "activation-patching",
        {"layer_results": [{"layer_index": 14, "recovery": 1.0}]},
    )

    assert assessment["severity"] == "ok"


def _flagged_result(metric_name: str) -> dict[str, object]:
    return {
        "flagged_layer_count": 1,
        "flagged_layers": [
            {
                "layer_name": "model.layers.0",
                "norm_ratio": 5.3,
                "nonfinite_count": 0,
                "top_parameters": [
                    {
                        "name": "model.layers.0.self_attn.k_proj.bias",
                        metric_name: 1102.7,
                    },
                ],
            },
        ],
    }
