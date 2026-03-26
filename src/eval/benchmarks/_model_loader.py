"""Shared model loading utilities for evaluation benchmarks.

Loads a trained Crucible model and tokenizer from disk for inference,
providing helpers for logit computation, perplexity, and text generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import CrucibleBenchmarkError, CrucibleDependencyError


@dataclass
class EvalModel:
    """Loaded model and tokenizer ready for benchmark evaluation."""

    model: Any
    tokenizer: Any
    torch_module: Any
    device: Any
    max_token_length: int


def load_eval_model(model_path: str) -> EvalModel:
    """Load a model for evaluation.

    Supports both Crucible checkpoints (.pt) and HuggingFace model directories.

    Args:
        model_path: Path to model checkpoint or HuggingFace model directory.

    Returns:
        EvalModel with model in eval mode.
    """
    torch_module = _import_torch()
    from serve.hf_model_loader import is_huggingface_model_id

    if is_huggingface_model_id(model_path):
        return _load_hf_eval_model(torch_module, model_path)
    # Check if this is a merged HF model (e.g. from LoRA training on a HF base).
    hf_base = _detect_hf_base_model(model_path)
    if hf_base:
        return _load_hf_eval_model(torch_module, hf_base, weights_path=model_path)
    return _load_crucible_eval_model(torch_module, model_path)


def _load_hf_eval_model(
    torch_module: Any,
    model_path: str,
    weights_path: str | None = None,
) -> EvalModel:
    """Load a HuggingFace model for evaluation."""
    from serve.device_selection import resolve_execution_device
    from serve.hf_model_loader import (
        load_huggingface_model,
        load_huggingface_tokenizer,
        _make_logits_wrapper,
    )

    device = resolve_execution_device(torch_module)
    hf_model = load_huggingface_model(model_path, weights_path, device)
    model = _make_logits_wrapper(hf_model)
    model.eval()

    hf_tokenizer = load_huggingface_tokenizer(model_path)
    tokenizer = _HfTokenizerAdapter(hf_tokenizer)

    max_token_length = getattr(hf_tokenizer, "model_max_length", 1024)
    if max_token_length > 100_000:
        max_token_length = 1024

    return EvalModel(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        max_token_length=max_token_length,
    )


def _load_crucible_eval_model(torch_module: Any, model_path: str) -> EvalModel:
    """Load a Crucible checkpoint model for evaluation."""
    from serve.chat_option_resolver import (
        resolve_chat_model_vocab_size,
        resolve_chat_training_options,
    )
    from serve.architecture_loader import load_training_model
    from serve.device_selection import resolve_execution_device
    from serve.model_weights import load_initial_weights, read_model_state_dict
    from serve.training_metadata import load_tokenizer

    device = resolve_execution_device(torch_module)
    model_state = read_model_state_dict(torch_module, model_path, device)

    from core.chat_types import ChatOptions
    chat_opts = ChatOptions(model_path=model_path, prompt="")
    training_options = resolve_chat_training_options(chat_opts, model_state)

    tokenizer = load_tokenizer(model_path)
    if tokenizer is None:
        raise CrucibleBenchmarkError(
            f"No tokenizer found beside model at {model_path}. "
            "Ensure vocab.json exists in the model directory."
        )

    vocab_size = resolve_chat_model_vocab_size(
        tokenizer.vocabulary, model_state, training_options,
    )
    model = load_training_model(torch_module, training_options, vocab_size)
    model = model.to(device)
    load_initial_weights(torch_module, model, model_path, device)
    model.eval()

    return EvalModel(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        max_token_length=training_options.max_token_length,
    )


def compute_logits(eval_model: EvalModel, text: str) -> Any:
    """Run a forward pass and return logits for the last token.

    Args:
        eval_model: Loaded model context.
        text: Input text to tokenize and feed.

    Returns:
        Logits tensor of shape [vocab_size] for the last position.
    """
    torch = eval_model.torch_module
    ids = eval_model.tokenizer.encode(text, eval_model.max_token_length)
    if not ids:
        return torch.zeros(1)
    tensor = torch.tensor([ids], dtype=torch.long).to(eval_model.device)
    with torch.no_grad():
        output = eval_model.model(tensor)
    logits = output.logits if hasattr(output, "logits") else output
    return logits[0, -1, :]


def compute_sequence_loss(eval_model: EvalModel, text: str) -> float:
    """Compute average cross-entropy loss over a text sequence.

    Lower loss means the model assigns higher probability to the text.
    Used for perplexity-based scoring in multiple-choice benchmarks.

    Args:
        eval_model: Loaded model context.
        text: Full text to score.

    Returns:
        Average cross-entropy loss (lower = better fit).
    """
    torch = eval_model.torch_module
    ids = eval_model.tokenizer.encode(text, eval_model.max_token_length)
    if len(ids) < 2:
        return float("inf")
    tensor = torch.tensor([ids], dtype=torch.long).to(eval_model.device)
    with torch.no_grad():
        output = eval_model.model(tensor)
    logits = output.logits if hasattr(output, "logits") else output
    shift_logits = logits[0, :-1, :]
    shift_labels = tensor[0, 1:]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits, shift_labels)
    return float(loss.item())


def compute_completion_loss(eval_model: EvalModel, prompt: str, completion: str) -> float:
    """Compute average cross-entropy loss over completion tokens only.

    Tokenizes prompt + completion together but only computes loss on the
    completion portion.  This avoids the shared prompt dominating the loss
    when comparing multiple completions for the same prompt.

    Args:
        eval_model: Loaded model context.
        prompt: Context / prompt text (not scored).
        completion: Completion text to score.

    Returns:
        Average cross-entropy loss over the completion tokens.
    """
    torch = eval_model.torch_module
    prompt_ids = eval_model.tokenizer.encode(prompt, eval_model.max_token_length)
    full_ids = eval_model.tokenizer.encode(
        prompt + completion, eval_model.max_token_length,
    )
    # Number of tokens belonging to the completion
    completion_len = len(full_ids) - len(prompt_ids)
    if completion_len < 1:
        return float("inf")
    tensor = torch.tensor([full_ids], dtype=torch.long).to(eval_model.device)
    with torch.no_grad():
        output = eval_model.model(tensor)
    logits = output.logits if hasattr(output, "logits") else output
    # Only score the completion positions: logits at [prompt_len-1 .. end-1]
    # predict tokens at [prompt_len .. end]
    start = max(len(prompt_ids) - 1, 0)
    shift_logits = logits[0, start:-1, :]
    shift_labels = tensor[0, start + 1:]
    if shift_logits.shape[0] == 0:
        return float("inf")
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits, shift_labels)
    return float(loss.item())


def generate_text(
    eval_model: EvalModel,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate text autoregressively using greedy decoding.

    Args:
        eval_model: Loaded model context.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text (excluding prompt).
    """
    torch = eval_model.torch_module
    context_ids = eval_model.tokenizer.encode(prompt, eval_model.max_token_length)
    generated: list[int] = []

    eos_id = getattr(eval_model.tokenizer, "eos_token_id", 0)

    for _ in range(max_new_tokens):
        window = context_ids[-(eval_model.max_token_length):]
        tensor = torch.tensor([window], dtype=torch.long).to(eval_model.device)
        with torch.no_grad():
            output = eval_model.model(tensor)
        logits = output.logits if hasattr(output, "logits") else output
        next_id = int(logits[0, -1, :].argmax().item())
        if next_id == 0 or next_id == eos_id:
            break
        context_ids.append(next_id)
        generated.append(next_id)

    return eval_model.tokenizer.decode(generated)


class _HfTokenizerAdapter:
    """Adapt a HuggingFace AutoTokenizer to the ChatTokenizer protocol."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tokenizer = hf_tokenizer
        self.vocabulary: dict[str, int] = dict(hf_tokenizer.get_vocab())
        self.eos_token_id: int = hf_tokenizer.eos_token_id or 0

    def encode(self, text: str, max_token_length: int) -> list[int]:
        ids: list[int] = self._tokenizer.encode(text)
        if len(ids) > max_token_length:
            return ids[:max_token_length]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids))


def _detect_hf_base_model(model_path: str) -> str | None:
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
        resolved = _resolve_remote_hf_id(value)
        if resolved:
            return resolved
    return None


def _resolve_remote_hf_id(remote_path: str) -> str | None:
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
    # Try unsanitizing first underscore → slash (e.g. "openai-community_gpt2" → "openai-community/gpt2")
    idx = basename.find("_")
    if idx > 0:
        candidate = basename[:idx] + "/" + basename[idx + 1:]
        if is_huggingface_model_id(candidate):
            return candidate
    # Fall back to basename directly (e.g. "gpt2")
    if is_huggingface_model_id(basename):
        return basename
    return None


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
        return torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "Evaluation benchmarks require torch. Install torch to run evaluations."
        ) from error
