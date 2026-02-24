"""Reward model scoring and loading for RLHF training.

This module provides reward model abstractions for scoring policy responses,
loading external reward models, and building reward heads for training.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Protocol

from core.errors import ForgeRlhfError


class RewardScorer(Protocol):
    """Protocol for reward scoring implementations."""

    def score(self, prompt: str, response: str) -> float:
        """Score a prompt-response pair.

        Args:
            prompt: Input prompt text.
            response: Generated response text.

        Returns:
            Scalar reward score.
        """
        ...


def load_external_reward_model(
    torch_module: Any,
    model: Any,
    path: str,
    device: Any,
) -> Any:
    """Load reward model weights from a saved checkpoint.

    Args:
        torch_module: Imported torch module.
        model: Reward model architecture to load weights into.
        path: Path to saved reward model checkpoint.
        device: Torch device for the model.

    Returns:
        Loaded reward model in eval mode.

    Raises:
        ForgeRlhfError: If checkpoint cannot be loaded.
    """
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise ForgeRlhfError(
            f"Reward model not found at {resolved_path}. "
            "Provide a valid --reward-model-path."
        )
    try:
        state_dict = torch_module.load(
            str(resolved_path), map_location=device,
        )
    except (OSError, RuntimeError) as error:
        raise ForgeRlhfError(
            f"Failed to load reward model from {resolved_path}: {error}."
        ) from error
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_reward_scores(
    torch_module: Any,
    reward_model: Any,
    input_ids: Any,
    device: Any,
) -> Any:
    """Compute reward scores for batched token sequences.

    Args:
        torch_module: Imported torch module.
        reward_model: Reward model producing scalar outputs.
        input_ids: Batched input token ids tensor [batch, seq_len].
        device: Torch device.

    Returns:
        Reward scores tensor of shape [batch].
    """
    with torch_module.no_grad():
        logits = reward_model(input_ids.to(device))
    if logits.dim() > 1:
        return logits[:, -1].squeeze(-1)
    return logits.squeeze(-1)


def build_reward_head(torch_module: Any, hidden_dim: int) -> Any:
    """Build a simple linear reward head for training.

    The reward head maps from hidden representations to scalar scores.

    Args:
        torch_module: Imported torch module.
        hidden_dim: Hidden dimension of the base model.

    Returns:
        torch.nn.Module with linear projection to scalar output.
    """
    nn = torch_module.nn
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def create_reference_policy(torch_module: Any, model: Any) -> Any:
    """Create a frozen deep copy of the policy model as reference.

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


def build_reward_model_from_base(
    torch_module: Any,
    base_model: Any,
    hidden_dim: int,
) -> Any:
    """Build a reward model by combining base model with reward head.

    Args:
        torch_module: Imported torch module.
        base_model: Base language model for feature extraction.
        hidden_dim: Hidden dimension for the reward head.

    Returns:
        Combined reward model module.
    """
    reward_head = build_reward_head(torch_module, hidden_dim)
    base_copy = copy.deepcopy(base_model)
    return _RewardModelWrapper(torch_module, base_copy, reward_head)


class _RewardModelWrapper:
    """Wraps a base model with a reward scoring head."""

    def __init__(
        self, torch_module: Any, base: Any, head: Any,
    ) -> None:
        self._torch = torch_module
        self._base = base
        self._head = head
        self._module = torch_module.nn.ModuleList([base, head])

    def __call__(self, input_ids: Any) -> Any:
        """Forward pass producing scalar reward scores."""
        base_output = self._base(input_ids)
        if base_output.dim() == 3:
            last_hidden = base_output[:, -1, :]
        else:
            last_hidden = base_output
        return self._head(last_hidden).squeeze(-1)

    def parameters(self) -> Any:
        """Return all trainable parameters."""
        return self._module.parameters()

    def to(self, device: Any) -> "_RewardModelWrapper":
        """Move model to device."""
        self._module.to(device)
        return self

    def state_dict(self) -> Any:
        """Return combined state dict."""
        return self._module.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        """Load combined state dict."""
        self._module.load_state_dict(state_dict)

    def eval(self) -> "_RewardModelWrapper":
        """Set model to eval mode."""
        self._module.eval()
        return self

    def train(self, mode: bool = True) -> "_RewardModelWrapper":
        """Set model to train mode."""
        self._module.train(mode)
        return self
