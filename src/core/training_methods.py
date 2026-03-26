"""Centralized training method definitions.

Single source of truth for all training method identifiers, display names,
and dispatch metadata. Mirrors the TypeScript TrainingMethod type in
studio-app/src/types/training.ts.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Literal, get_origin, get_args, get_type_hints, Union

from core.distillation_types import DistillationOptions
from core.domain_adaptation_types import DomainAdaptationOptions
from core.dpo_types import DpoOptions
from core.grpo_types import GrpoOptions
from core.kto_types import KtoOptions
from core.lora_types import LoraTrainingOptions
from core.multimodal_types import MultimodalOptions
from core.orpo_types import OrpoOptions
from core.qlora_types import QloraOptions
from core.rlhf_types import RlhfOptions
from core.rlvr_types import RlvrOptions
from core.sft_types import SftOptions
from core.training_types import TrainingOptions, TrainingRunResult

TrainingMethod = Literal[
    "train",
    "sft",
    "dpo-train",
    "rlhf-train",
    "lora-train",
    "distill",
    "domain-adapt",
    "grpo-train",
    "qlora-train",
    "kto-train",
    "orpo-train",
    "multimodal-train",
    "rlvr-train",
]

ALL_TRAINING_METHODS: tuple[str, ...] = (
    "train", "sft", "dpo-train", "rlhf-train", "lora-train",
    "distill", "domain-adapt", "grpo-train", "qlora-train",
    "kto-train", "orpo-train", "multimodal-train", "rlvr-train",
)

TRAINING_METHOD_LABELS: dict[str, str] = {
    "train": "Basic Training",
    "sft": "SFT",
    "dpo-train": "DPO",
    "rlhf-train": "RLHF",
    "lora-train": "LoRA",
    "distill": "Distillation",
    "domain-adapt": "Domain Adaptation",
    "grpo-train": "GRPO",
    "qlora-train": "QLoRA",
    "kto-train": "KTO",
    "orpo-train": "ORPO",
    "multimodal-train": "Multimodal",
    "rlvr-train": "RLVR",
}

# Maps method ID -> (client_method_name, OptionsClass)
TRAINING_METHOD_DISPATCH: dict[str, tuple[str, type]] = {
    "train":            ("train",            TrainingOptions),
    "sft":              ("sft_train",        SftOptions),
    "dpo-train":        ("dpo_train",        DpoOptions),
    "rlhf-train":       ("rlhf_train",       RlhfOptions),
    "lora-train":       ("lora_train",       LoraTrainingOptions),
    "distill":          ("distill",          DistillationOptions),
    "domain-adapt":     ("domain_adapt",     DomainAdaptationOptions),
    "grpo-train":       ("grpo_train",       GrpoOptions),
    "qlora-train":      ("qlora_train",      QloraOptions),
    "kto-train":        ("kto_train",        KtoOptions),
    "orpo-train":       ("orpo_train",       OrpoOptions),
    "multimodal-train": ("multimodal_train", MultimodalOptions),
    "rlvr-train":       ("rlvr_train",       RlvrOptions),
}

# Maps method ID -> data path field name on its options class.
# Methods not listed here (train, distill, domain-adapt) use dataset records directly.
DATA_PATH_FIELDS: dict[str, str] = {
    "sft":              "sft_data_path",
    "dpo-train":        "dpo_data_path",
    "lora-train":       "lora_data_path",
    "qlora-train":      "qlora_data_path",
    "grpo-train":       "grpo_data_path",
    "kto-train":        "kto_data_path",
    "orpo-train":       "orpo_data_path",
    "multimodal-train": "multimodal_data_path",
    "rlvr-train":       "rlvr_data_path",
}


def _unwrap_optional(tp: Any) -> Any | None:
    """Return the inner type if *tp* is ``X | None``, else ``None``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return None


def _coerce_value(value: Any, target_type: Any) -> Any:
    """Coerce *value* to match *target_type* annotation."""
    if value is None:
        return None

    # Unwrap Optional[X] → coerce to X
    inner = _unwrap_optional(target_type)
    if inner is not None:
        return _coerce_value(value, inner)

    # tuple[str, ...] — accept comma-separated string or list
    if get_origin(target_type) is tuple:
        if isinstance(value, str):
            return tuple(s.strip() for s in value.split(","))
        if isinstance(value, list):
            return tuple(value)
        return value

    # Nested dataclass — recursively coerce dict
    if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
        return _coerce_dataclass_kwargs(target_type, value)

    # Primitive coercions from string
    if isinstance(value, str):
        if target_type is int:
            return int(float(value))
        if target_type is float:
            return float(value)
        if target_type is bool:
            return value.lower() in ("true", "1", "yes")

    # float → int (e.g. 3.0 from sweep)
    if isinstance(value, float) and target_type is int:
        return int(value)

    # list → tuple
    if target_type is tuple and isinstance(value, list):
        return tuple(value)

    return value


def _coerce_dataclass_kwargs(dc_class: type, kwargs: dict[str, Any]) -> Any:
    """Build a dataclass instance, coercing each value to its annotated type."""
    hints = get_type_hints(dc_class)
    valid_fields = set(dc_class.__dataclass_fields__)
    coerced: dict[str, Any] = {}
    for key, val in kwargs.items():
        if key not in valid_fields:
            continue
        coerced[key] = _coerce_value(val, hints[key])
    return dc_class(**coerced)


# Flat key → (nested_field, inner_key) for methods with nested config dataclasses.
# Allows the UI/remote to send e.g. {"lora_rank": "64"} instead of
# {"lora_config": {"rank": 64}}.
_FLAT_KEY_CONFIGS: dict[str, dict[str, tuple[str, str]]] = {
    "lora-train": {
        "lora_rank": ("lora_config", "rank"),
        "lora_alpha": ("lora_config", "alpha"),
        "lora_dropout": ("lora_config", "dropout"),
        "lora_target_modules": ("lora_config", "target_modules"),
    },
    "rlhf-train": {
        "reward_model_path": ("reward_config", "reward_model_path"),
        "train_reward_model": ("reward_config", "train_reward_model"),
        "preference_data_path": ("reward_config", "preference_data_path"),
        "clip_epsilon": ("ppo_config", "clip_epsilon"),
        "ppo_epochs": ("ppo_config", "ppo_epochs"),
        "entropy_coeff": ("ppo_config", "entropy_coeff"),
        "value_loss_coeff": ("ppo_config", "value_loss_coeff"),
        "gamma": ("ppo_config", "gamma"),
        "lam": ("ppo_config", "lam"),
    },
}


def _nest_flat_keys(method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Restructure flat keys into nested dicts per ``_FLAT_KEY_CONFIGS``."""
    mapping = _FLAT_KEY_CONFIGS.get(method)
    if not mapping:
        return dict(kwargs)
    result: dict[str, Any] = {}
    for key, val in kwargs.items():
        target = mapping.get(key)
        if target:
            nested_field, inner_key = target
            result.setdefault(nested_field, {})[inner_key] = val
        else:
            result[key] = val
    return result


def dispatch_training(
    client: Any,
    method: str,
    kwargs: dict[str, Any],
) -> TrainingRunResult:
    """Build method-specific options and call the correct client method.

    Handles flat-key nesting (e.g. ``lora_rank`` → ``lora_config.rank``)
    and type coercion (e.g. ``"64"`` → ``64``) so that both local CLI
    and remote JSON pipelines produce identical Options objects.

    Args:
        client: CrucibleClient instance.
        method: Training method identifier.
        kwargs: Keyword arguments for the options dataclass.

    Returns:
        Training run result from the executed method.

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in TRAINING_METHOD_DISPATCH:
        raise ValueError(
            f"Unknown training method '{method}'. "
            f"Valid methods: {', '.join(ALL_TRAINING_METHODS)}"
        )
    client_method_name, options_class = TRAINING_METHOD_DISPATCH[method]
    nested = _nest_flat_keys(method, kwargs)
    options = _coerce_dataclass_kwargs(options_class, nested)
    return getattr(client, client_method_name)(options)
