"""DPO reference model management.

This module creates and manages frozen reference models used by the
DPO training loop to compute baseline log probabilities for the
preference optimization objective.
"""

from __future__ import annotations

import copy
from typing import Any

from serve.dpo_loss import compute_log_probs_from_logits


def create_reference_model(
    torch_module: Any,
    model: Any,
) -> Any:
    """Create a frozen deep copy of the policy model as reference.

    The reference model is used to compute baseline log probabilities.
    All parameters are detached and set to non-trainable.

    Args:
        torch_module: Imported torch module.
        model: Policy model to copy.

    Returns:
        Frozen copy of the model in eval mode.
    """
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model


def load_reference_model(
    torch_module: Any,
    model_template: Any,
    reference_model_path: str,
    device: Any,
) -> Any:
    """Load reference model weights from a saved checkpoint.

    Args:
        torch_module: Imported torch module.
        model_template: Model architecture to load weights into.
        reference_model_path: Path to saved model weights.
        device: Torch device for the model.

    Returns:
        Frozen reference model loaded from checkpoint.
    """
    ref_model = copy.deepcopy(model_template)
    state_dict = torch_module.load(reference_model_path, map_location=device)
    ref_model.load_state_dict(state_dict)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model


def compute_reference_log_probs(
    torch_module: Any,
    ref_model: Any,
    input_ids: Any,
    labels: Any,
    prompt_length: int,
) -> Any:
    """Compute log probabilities from the reference model.

    Runs forward pass through the frozen reference model and extracts
    per-response log probabilities for DPO loss computation.

    Args:
        torch_module: Imported torch module.
        ref_model: Frozen reference model.
        input_ids: Input token ids tensor.
        labels: Target label tensor.
        prompt_length: Number of prompt tokens for masking.

    Returns:
        Summed log probability tensor of shape (batch,).
    """
    with torch_module.no_grad():
        if getattr(ref_model, "_is_hf_logits_wrapper", False):
            attention_mask = (input_ids != 0).long()
            ref_logits = ref_model(input_ids, attention_mask=attention_mask)
        else:
            ref_logits = ref_model(input_ids)
    return compute_log_probs_from_logits(
        torch_module, ref_logits, labels, prompt_length
    )
