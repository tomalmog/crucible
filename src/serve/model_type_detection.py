"""Shared utilities for detecting model types (HF vs Crucible, LoRA detection).

Used by chat_runner, eval/_model_loader, and interp_model_loader to avoid
triplicated detection logic.
"""

from __future__ import annotations

from typing import Any


def detect_hf_base_model(model_path: str) -> str | None:
    """Check if model_path has a training_config with a HuggingFace base model.

    Checks both ``base_model_path`` (used by LoRA/SFT/DPO) and
    ``initial_weights_path`` (used by QLoRA) so that all fine-tuned
    HuggingFace models are detected regardless of training method.

    Handles remote cluster paths (e.g. ``/u201/.../openai-community_gpt2``)
    by extracting the basename and trying it as a HuggingFace model ID,
    including unsanitized variants (``org_model`` → ``org/model``).
    """
    from serve.hf_model_loader import is_huggingface_model_id
    from serve.training_metadata import load_training_config

    config = load_training_config(model_path)
    if not config:
        return None
    for key in ("base_model_path", "initial_weights_path"):
        value = str(config.get(key, "") or "")
        if not value:
            continue
        if is_huggingface_model_id(value):
            return value
        resolved = resolve_remote_hf_id(value)
        if resolved:
            return resolved
    return None


def resolve_remote_hf_id(remote_path: str) -> str | None:
    """Try to recover a HuggingFace model ID from a remote cluster path.

    Remote paths like ``/u201/.../openai-community_gpt2`` have the model
    name as the basename, sanitized by ``sanitize_remote_name`` which
    replaces ``/`` with ``_``.  Try unsanitizing first (``org_model`` →
    ``org/model``) since that's the most common HF ID pattern, then
    fall back to the raw basename (e.g. ``gpt2``).
    """
    from pathlib import Path as _Path
    from serve.hf_model_loader import is_huggingface_model_id

    basename = _Path(remote_path).name
    if not basename:
        return None
    # Try each underscore position as a potential org/model separator.
    # sanitize_remote_name replaces "/" with "_", so "org/model" becomes
    # "org_model". For multi-underscore names like "meta-llama_Llama-2_7b",
    # try "meta-llama/Llama-2_7b", then "meta-llama_Llama-2/7b", etc.
    for idx in range(len(basename)):
        if basename[idx] == "_" and idx > 0 and idx < len(basename) - 1:
            candidate = basename[:idx] + "/" + basename[idx + 1:]
            if is_huggingface_model_id(candidate):
                return candidate
    # Fall back to basename directly (e.g. "gpt2")
    if is_huggingface_model_id(basename):
        return basename
    return None
