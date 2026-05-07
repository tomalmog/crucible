"""Curated model health check suite definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

HealthSuiteId = Literal["standard", "deep", "supervised", "targeted"]
HealthCheckId = Literal[
    "weight-norms",
    "activation-norms",
    "gradient-norms",
    "logit-lens",
    "activation-pca",
    "activation-patching",
    "linear-probe",
    "linear-probe-layers",
]


@dataclass(frozen=True)
class HealthCheckDefinition:
    """A single diagnostic included in a model health suite."""

    check_id: HealthCheckId
    label: str
    reason: str
    requires_dataset: bool = False
    requires_label: bool = False
    requires_probe: bool = False
    requires_contrast: bool = False
    supports_layers: bool = False
    is_expensive: bool = False


@dataclass(frozen=True)
class ModelHealthCheckOptions:
    """Inputs shared by diagnostics in a model health suite."""

    model_path: str
    dataset_name: str
    probe_text: str
    clean_text: str
    corrupted_text: str
    label_field: str = ""
    max_samples: int = 300
    base_model: str = ""
    output_dir: str = "./outputs/model-health"
    check_ids: tuple[str, ...] = ()
    layer_indices: str = ""


@dataclass(frozen=True)
class ModelHealthCheckCommand:
    """Executable diagnostic command for a model health suite."""

    config: dict[str, object]
    label: str
    tool_name: HealthCheckId
    remote_job_type: str
    args: dict[str, object]

    def args_json(self) -> str:
        """Serialize arguments for the local run_interp MCP tool."""
        return json.dumps(self.args)

    def remote_args_json(self, model_path: str) -> str:
        """Serialize arguments for remote interpretability submission."""
        return json.dumps({"model_path": model_path, **self.args})


WEIGHT_NORMS = HealthCheckDefinition(
    "weight-norms", "Weight stability", "Scans layers for weight norm spikes or invalid values.",
    supports_layers=True,
)
ACTIVATION_NORMS = HealthCheckDefinition(
    "activation-norms", "Activation stability", "Checks layer activation norms on calibration prompts.",
    requires_dataset=True, supports_layers=True,
)
GRADIENT_NORMS = HealthCheckDefinition(
    "gradient-norms", "Gradient stability", "Backprops a calibration objective to find exploding layer gradients.",
    requires_dataset=True, supports_layers=True, is_expensive=True,
)
LOGIT_LENS = HealthCheckDefinition(
    "logit-lens", "Prediction trace", "Shows how next-token predictions change across model layers.",
    requires_probe=True, supports_layers=True,
)
ACTIVATION_PCA = HealthCheckDefinition(
    "activation-pca", "Representation map",
    "Projects activations for calibration samples to catch drift or shortcut clusters.",
    requires_dataset=True, supports_layers=True,
)
ACTIVATION_PATCHING = HealthCheckDefinition(
    "activation-patching", "Causal contrast",
    "Tests which activations drive a behavior difference between two prompts.",
    requires_contrast=True, supports_layers=True,
)
LINEAR_PROBE = HealthCheckDefinition(
    "linear-probe", "Label separability", "Checks whether supervised labels are encoded in frozen representations.",
    requires_dataset=True, requires_label=True, supports_layers=True,
)
LINEAR_PROBE_LAYERS = HealthCheckDefinition(
    "linear-probe-layers", "Layer-wise label probe",
    "Runs linear probes across layers to locate where a label becomes separable.",
    requires_dataset=True, requires_label=True, supports_layers=True, is_expensive=True,
)

CHECK_REGISTRY: dict[str, HealthCheckDefinition] = {
    check.check_id: check
    for check in (
        WEIGHT_NORMS,
        ACTIVATION_NORMS,
        GRADIENT_NORMS,
        LOGIT_LENS,
        ACTIVATION_PCA,
        ACTIVATION_PATCHING,
        LINEAR_PROBE,
        LINEAR_PROBE_LAYERS,
    )
}

STANDARD_CHECKS = (WEIGHT_NORMS, ACTIVATION_NORMS, ACTIVATION_PCA, ACTIVATION_PATCHING)
DEEP_CHECKS = (*STANDARD_CHECKS, GRADIENT_NORMS, LOGIT_LENS)
SUPERVISED_CHECKS = (WEIGHT_NORMS, ACTIVATION_NORMS, LINEAR_PROBE_LAYERS, ACTIVATION_PCA)


def available_health_checks() -> tuple[HealthCheckDefinition, ...]:
    """Return every registered health check."""
    return tuple(CHECK_REGISTRY.values())


def health_suite_checks(suite_id: str, options: ModelHealthCheckOptions | None = None) -> tuple[HealthCheckDefinition, ...]:
    """Return the diagnostics included in a named health suite."""
    if suite_id == "targeted" and options is not None:
        return tuple(_definition_for(check_id) for check_id in options.check_ids)
    if suite_id == "deep":
        return DEEP_CHECKS
    if suite_id == "supervised":
        return SUPERVISED_CHECKS
    return STANDARD_CHECKS


def health_suite_title(suite_id: str) -> str:
    """Return the user-facing title for a health suite."""
    if suite_id == "deep":
        return "Deep Stability Assessment"
    if suite_id == "supervised":
        return "Supervised Behavior Check"
    if suite_id == "targeted":
        return "Targeted Health Investigation"
    return "Standard Health Check"


def validate_model_health_options(
    suite_id: str,
    options: ModelHealthCheckOptions,
) -> tuple[str, ...]:
    """Return missing fields that prevent selected checks from running."""
    checks = health_suite_checks(suite_id, options)
    missing: list[str] = []
    if not options.model_path.strip():
        missing.append("model_path")
    if suite_id == "targeted" and not checks:
        missing.append("checks")
    if any(check.requires_dataset for check in checks) and not options.dataset_name.strip():
        missing.append("dataset_name")
    if any(check.requires_probe for check in checks) and not options.probe_text.strip():
        missing.append("probe_text")
    if any(check.requires_contrast for check in checks):
        if not options.clean_text.strip():
            missing.append("clean_text")
        if not options.corrupted_text.strip():
            missing.append("corrupted_text")
    if any(check.requires_label for check in checks) and not options.label_field.strip():
        missing.append("label_field")
    return tuple(dict.fromkeys(missing))


def build_model_health_check_commands(
    suite_id: str,
    options: ModelHealthCheckOptions,
) -> tuple[ModelHealthCheckCommand, ...]:
    """Build the executable diagnostics for a model health suite."""
    missing = validate_model_health_options(suite_id, options)
    if missing:
        raise ValueError(f"Missing required model health fields: {', '.join(missing)}")
    return tuple(_build_command(check, options) for check in health_suite_checks(suite_id, options))


def build_model_health_suite_args(suite_id: str, options: ModelHealthCheckOptions) -> dict[str, object]:
    """Build remote method args for one suite-level health job."""
    payload: dict[str, object] = {
        "model_path": options.model_path,
        "dataset_name": options.dataset_name,
        "suite": suite_id,
        "checks": ",".join(options.check_ids),
        "layer_indices": options.layer_indices,
        "probe_text": options.probe_text,
        "clean_text": options.clean_text,
        "corrupted_text": options.corrupted_text,
        "label_field": options.label_field,
        "max_samples": options.max_samples,
        "output_dir": options.output_dir,
    }
    if options.base_model.strip():
        payload["base_model"] = options.base_model
    return payload


def build_model_health_suite_config(suite_id: str, options: ModelHealthCheckOptions) -> dict[str, object]:
    """Build UI config for one suite-level health report."""
    checks = health_suite_checks(suite_id, options)
    return {
        "page": "interpretability",
        "tab": "health",
        "workflow": "model-health-check",
        "suiteId": suite_id,
        "suiteTitle": health_suite_title(suite_id),
        "healthChecks": [{"id": check.check_id, "label": check.label, "why": check.reason} for check in checks],
        "modelPath": options.model_path,
        "dataset": options.dataset_name,
        "probeText": options.probe_text,
        "cleanText": options.clean_text,
        "corruptedText": options.corrupted_text,
        "labelField": options.label_field,
        "maxSamples": options.max_samples,
        "baseModel": options.base_model,
        "checks": list(options.check_ids),
        "layerIndices": options.layer_indices,
    }


def _build_command(check: HealthCheckDefinition, options: ModelHealthCheckOptions) -> ModelHealthCheckCommand:
    return ModelHealthCheckCommand(
        config=_build_config(check, options),
        label=f"Model Health · {check.label}",
        tool_name=check.check_id,
        remote_job_type=_remote_job_type(check.check_id),
        args=_build_args(check, options),
    )


def _build_args(check: HealthCheckDefinition, options: ModelHealthCheckOptions) -> dict[str, object]:
    args: dict[str, object] = {
        **_build_config(check, options),
        "output_dir": _check_output_dir(options.output_dir, check.check_id),
        "dataset_name": options.dataset_name,
        "max_samples": options.max_samples,
        "layer_indices": options.layer_indices,
    }
    if options.base_model.strip():
        args["base_model"] = options.base_model
    if check.check_id == "logit-lens":
        args.update({"input_text": options.probe_text, "top_k": 5})
    if check.check_id == "activation-patching":
        args.update({"clean_text": options.clean_text, "corrupted_text": options.corrupted_text, "metric": "logit_diff"})
    if check.requires_label:
        args.update({"label_field": options.label_field, "epochs": 8, "learning_rate": 0.001})
    return args


def _build_config(check: HealthCheckDefinition, options: ModelHealthCheckOptions) -> dict[str, object]:
    return {
        "page": "interpretability",
        "tab": "health",
        "workflow": "model-health-check",
        "healthCheckId": check.check_id,
        "healthCheckLabel": check.label,
        "healthCheckReason": check.reason,
        "modelPath": options.model_path,
        "dataset": options.dataset_name,
    }


def _definition_for(check_id: str) -> HealthCheckDefinition:
    check = CHECK_REGISTRY.get(check_id)
    if check is None:
        raise ValueError(f"Unknown health check: {check_id}")
    return check


def _check_output_dir(output_dir: str, check_id: HealthCheckId) -> str:
    suffix = check_id.replace("activation-patching", "causal-contrast")
    return f"{output_dir.rstrip('/')}/{suffix}"


def _remote_job_type(check_id: HealthCheckId) -> str:
    if check_id == "activation-patching":
        return "activation-patch"
    return check_id
