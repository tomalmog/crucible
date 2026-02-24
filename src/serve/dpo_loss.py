"""DPO loss computation for preference optimization.

This module implements the Direct Preference Optimization loss function
that uses log-probability ratios between policy and reference models
to optimize for human preferences without explicit reward modeling.
"""

from __future__ import annotations

from typing import Any

from serve.dpo_tokenization import IGNORE_INDEX


def compute_log_probs_from_logits(
    torch_module: Any,
    logits: Any,
    labels: Any,
    prompt_length: int,
) -> Any:
    """Compute per-token log probabilities from model logits.

    Extracts log probabilities only for response tokens (after prompt),
    summing them into a scalar log probability for the full response.

    Args:
        torch_module: Imported torch module.
        logits: Model output logits of shape (batch, seq_len, vocab).
        labels: Target labels of shape (batch, seq_len).
        prompt_length: Number of prompt tokens to skip.

    Returns:
        Summed log probability tensor of shape (batch,).
    """
    log_probs = torch_module.nn.functional.log_softmax(logits, dim=-1)
    batch_size = labels.shape[0]
    seq_len = labels.shape[1]
    gathered = torch_module.zeros(batch_size, seq_len, device=logits.device)
    for b in range(batch_size):
        for t in range(seq_len):
            if labels[b, t] != IGNORE_INDEX and t >= prompt_length:
                gathered[b, t] = log_probs[b, t, labels[b, t]]
    response_mask = (labels != IGNORE_INDEX).float()
    if prompt_length < seq_len:
        response_mask[:, :prompt_length] = 0.0
    return (gathered * response_mask).sum(dim=-1)


def compute_dpo_loss(
    torch_module: Any,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    beta: float,
    label_smoothing: float,
) -> Any:
    """Compute DPO loss from policy and reference log probabilities.

    Loss = -log_sigmoid(beta * (log_pi_chosen - log_ref_chosen
                                - log_pi_rejected + log_ref_rejected))

    With label smoothing, the loss becomes a mixture between the
    standard DPO loss and a flipped version.

    Args:
        torch_module: Imported torch module.
        policy_chosen_logps: Policy log-probs for chosen responses.
        policy_rejected_logps: Policy log-probs for rejected responses.
        ref_chosen_logps: Reference log-probs for chosen responses.
        ref_rejected_logps: Reference log-probs for rejected responses.
        beta: DPO temperature parameter controlling preference strength.
        label_smoothing: Smoothing factor in [0, 0.5) for robustness.

    Returns:
        Scalar DPO loss tensor.
    """
    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps
    logits_diff = beta * (chosen_rewards - rejected_rewards)
    if label_smoothing > 0.0:
        loss_preferred = -torch_module.nn.functional.logsigmoid(logits_diff)
        loss_dispreferred = -torch_module.nn.functional.logsigmoid(-logits_diff)
        loss = (
            (1.0 - label_smoothing) * loss_preferred
            + label_smoothing * loss_dispreferred
        )
    else:
        loss = -torch_module.nn.functional.logsigmoid(logits_diff)
    return loss.mean()
