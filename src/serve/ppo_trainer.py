"""PPO training loop for RLHF policy optimization.

This module implements the Proximal Policy Optimization algorithm for
updating language model policies using reward model scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.rlhf_types import PpoConfig


@dataclass(frozen=True)
class PpoEpochResult:
    """Result from one PPO training epoch.

    Attributes:
        epoch: One-based epoch index.
        policy_loss: Average policy loss for the epoch.
        value_loss: Average value function loss for the epoch.
        mean_reward: Average reward score across generated responses.
    """

    epoch: int
    policy_loss: float
    value_loss: float
    mean_reward: float


def run_ppo_epoch(
    torch_module: Any,
    policy_model: Any,
    reward_model: Any,
    ref_model: Any,
    prompts: Any,
    ppo_config: PpoConfig,
    optimizer: Any,
    device: Any,
    epoch: int,
) -> PpoEpochResult:
    """Run one PPO training epoch over a batch of prompts.

    Args:
        torch_module: Imported torch module.
        policy_model: Trainable policy model.
        reward_model: Reward model for scoring responses.
        ref_model: Frozen reference policy for KL penalty.
        prompts: Batched prompt token ids [batch, seq_len].
        ppo_config: PPO hyperparameters.
        optimizer: Torch optimizer for policy model.
        device: Torch device.
        epoch: One-based epoch index.

    Returns:
        PpoEpochResult with loss and reward metrics.
    """
    policy_model.train()
    responses = _generate_responses(torch_module, policy_model, prompts, device)
    rewards = _compute_rewards(torch_module, reward_model, prompts, responses, device)
    old_log_probs = _compute_log_probs(torch_module, policy_model, responses, device)
    values = _estimate_values(torch_module, policy_model, responses, device)
    advantages = compute_advantages(
        torch_module, rewards, values, ppo_config.gamma, ppo_config.lam,
    )
    total_policy_loss = 0.0
    total_value_loss = 0.0
    for _ppo_step in range(ppo_config.ppo_epochs):
        new_log_probs = _compute_log_probs(
            torch_module, policy_model, responses, device,
        )
        policy_loss = compute_ppo_loss(
            torch_module, new_log_probs, old_log_probs.detach(),
            advantages.detach(), ppo_config.clip_epsilon,
        )
        value_loss = _compute_value_loss(
            torch_module, policy_model, responses, rewards, device,
        )
        entropy = _compute_entropy(torch_module, new_log_probs)
        loss = (
            policy_loss
            + ppo_config.value_loss_coeff * value_loss
            - ppo_config.entropy_coeff * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    num_steps = max(ppo_config.ppo_epochs, 1)
    return PpoEpochResult(
        epoch=epoch,
        policy_loss=total_policy_loss / num_steps,
        value_loss=total_value_loss / num_steps,
        mean_reward=rewards.mean().item(),
    )


def compute_advantages(
    torch_module: Any,
    rewards: Any,
    values: Any,
    gamma: float,
    lam: float,
) -> Any:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        torch_module: Imported torch module.
        rewards: Reward scores tensor [batch].
        values: Value estimates tensor [batch].
        gamma: Discount factor.
        lam: GAE lambda parameter.

    Returns:
        Advantage estimates tensor [batch].
    """
    deltas = rewards - values
    advantages = torch_module.zeros_like(deltas)
    running_advantage = torch_module.tensor(0.0, device=deltas.device)
    for t in reversed(range(deltas.size(0))):
        running_advantage = deltas[t] + gamma * lam * running_advantage
        advantages[t] = running_advantage
    mean = advantages.mean()
    if advantages.numel() > 1:
        std = advantages.std() + 1e-8
    else:
        std = torch_module.tensor(1.0, device=advantages.device)
    return (advantages - mean) / std


def compute_ppo_loss(
    torch_module: Any,
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    clip_epsilon: float,
) -> Any:
    """Compute PPO clipped surrogate objective loss.

    Args:
        torch_module: Imported torch module.
        log_probs: Current policy log probabilities [batch].
        old_log_probs: Old policy log probabilities [batch].
        advantages: Advantage estimates [batch].
        clip_epsilon: PPO clipping range.

    Returns:
        Scalar PPO policy loss.
    """
    ratio = torch_module.exp(log_probs - old_log_probs)
    clipped_ratio = torch_module.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon,
    )
    surrogate_1 = ratio * advantages
    surrogate_2 = clipped_ratio * advantages
    return -torch_module.min(surrogate_1, surrogate_2).mean()


def _generate_responses(
    torch_module: Any,
    policy_model: Any,
    prompts: Any,
    device: Any,
) -> Any:
    """Generate responses from policy model given prompts.

    Simple greedy generation for structural PPO implementation.
    """
    with torch_module.no_grad():
        logits = policy_model(prompts.to(device))
    if logits.dim() == 3:
        return logits.argmax(dim=-1)
    return prompts


def _compute_log_probs(
    torch_module: Any,
    model: Any,
    sequences: Any,
    device: Any,
) -> Any:
    """Compute per-sequence log probabilities from model."""
    logits = model(sequences.to(device))
    if logits.dim() == 3:
        log_probs = torch_module.nn.functional.log_softmax(logits, dim=-1)
        gathered = torch_module.gather(
            log_probs[:, :-1, :], 2, sequences[:, 1:].unsqueeze(-1),
        )
        return gathered.squeeze(-1).sum(dim=-1)
    return logits.mean(dim=-1) if logits.dim() == 2 else logits


def _estimate_values(
    torch_module: Any,
    model: Any,
    sequences: Any,
    device: Any,
) -> Any:
    """Estimate values for sequences using the policy model."""
    with torch_module.no_grad():
        logits = model(sequences.to(device))
    if logits.dim() == 3:
        return logits[:, -1, :].mean(dim=-1)
    return logits.mean(dim=-1) if logits.dim() == 2 else logits


def _compute_value_loss(
    torch_module: Any,
    model: Any,
    sequences: Any,
    rewards: Any,
    device: Any,
) -> Any:
    """Compute value function MSE loss."""
    values = _estimate_values(torch_module, model, sequences, device)
    values_grad = model(sequences.to(device))
    if values_grad.dim() == 3:
        pred_values = values_grad[:, -1, :].mean(dim=-1)
    elif values_grad.dim() == 2:
        pred_values = values_grad.mean(dim=-1)
    else:
        pred_values = values_grad
    return torch_module.nn.functional.mse_loss(pred_values, rewards.detach())


def _compute_entropy(torch_module: Any, log_probs: Any) -> Any:
    """Compute entropy bonus from log probabilities."""
    return -(log_probs * torch_module.exp(log_probs)).mean()
