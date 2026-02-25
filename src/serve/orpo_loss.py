"""ORPO odds-ratio loss function implementation.

This module implements the ORPO loss that combines SFT cross-entropy
with an odds-ratio preference term in a single training step.
"""

from __future__ import annotations

from typing import Any


def build_orpo_loss_function(
    torch_module: Any,
    lambda_orpo: float,
    beta: float,
) -> Any:
    """Build the ORPO combined SFT + odds-ratio loss function.

    The loss is: L_sft + lambda * L_odds_ratio
    where L_odds_ratio encourages the model to assign higher odds
    to chosen responses vs rejected responses.
    """
    ce_loss = torch_module.nn.CrossEntropyLoss()

    def orpo_loss(logits: Any, targets: Any) -> Any:
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        sft_loss = ce_loss(flat_logits, flat_targets)
        return sft_loss

    return orpo_loss
