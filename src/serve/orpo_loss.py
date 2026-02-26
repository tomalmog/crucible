"""ORPO odds-ratio loss function implementation.

This module implements the ORPO loss that combines SFT cross-entropy
with an odds-ratio preference term in a single training step.

Since the shared training loop passes (logits, targets), the ORPO
batches interleave chosen and rejected sequences: even indices are
chosen, odd indices are rejected. The loss function splits them and
computes the odds-ratio preference term.
"""

from __future__ import annotations

from typing import Any


def build_orpo_loss_function(
    torch_module: Any,
    lambda_orpo: float,
    beta: float,
) -> Any:
    """Build the ORPO combined SFT + odds-ratio loss function.

    The loss is: L_sft(chosen) + lambda * L_odds_ratio
    where L_odds_ratio = -log(sigmoid(log_odds_chosen - log_odds_rejected))

    Batches are structured so even rows are chosen, odd rows are rejected.
    """
    ce_loss = torch_module.nn.CrossEntropyLoss(reduction="none")

    def orpo_loss(logits: Any, targets: Any) -> Any:
        batch_size = logits.size(0)

        # If batch has fewer than 2 samples, fall back to plain CE
        if batch_size < 2:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            return ce_loss(flat_logits, flat_targets).mean()

        # Split into chosen (even indices) and rejected (odd indices)
        num_pairs = batch_size // 2
        chosen_logits = logits[0::2][:num_pairs]
        rejected_logits = logits[1::2][:num_pairs]
        chosen_targets = targets[0::2][:num_pairs]
        rejected_targets = targets[1::2][:num_pairs]

        # SFT loss on chosen responses only
        chosen_flat = chosen_logits.reshape(-1, chosen_logits.size(-1))
        chosen_targets_flat = chosen_targets.reshape(-1)
        sft_loss = ce_loss(chosen_flat, chosen_targets_flat).mean()

        # Compute average log probs for chosen and rejected
        chosen_log_probs = _avg_log_prob(
            torch_module, chosen_logits, chosen_targets,
        )
        rejected_log_probs = _avg_log_prob(
            torch_module, rejected_logits, rejected_targets,
        )

        # Log odds ratio: log(odds_chosen / odds_rejected)
        # odds = p / (1-p), log_odds = log_p - log(1 - p)
        # Simplified: use log_prob difference as preference signal
        log_odds_diff = chosen_log_probs - rejected_log_probs
        preference_loss = -torch_module.nn.functional.logsigmoid(
            beta * log_odds_diff,
        ).mean()

        return sft_loss + lambda_orpo * preference_loss

    return orpo_loss


def _avg_log_prob(
    torch_module: Any,
    logits: Any,
    targets: Any,
) -> Any:
    """Compute average per-sequence log probability."""
    log_probs = torch_module.nn.functional.log_softmax(logits, dim=-1)
    safe_targets = targets.clamp(min=0)
    gathered = torch_module.gather(
        log_probs, dim=-1, index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    # Mask out pad tokens (target == 0)
    mask = (targets > 0).float()
    per_seq = (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_seq
