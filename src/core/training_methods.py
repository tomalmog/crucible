"""Centralized training method definitions.

Single source of truth for all training method identifiers, display names,
and dispatch metadata. Mirrors the TypeScript TrainingMethod type in
studio-app/src/types/training.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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


def dispatch_training(
    client: Any,
    method: str,
    kwargs: dict[str, Any],
) -> TrainingRunResult:
    """Build method-specific options and call the correct client method.

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
    valid_fields = {f for f in options_class.__dataclass_fields__}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    options = options_class(**filtered)
    return getattr(client, client_method_name)(options)
