"""KTO loss function implementation.

This module implements the Kahneman-Tversky Optimization loss
with asymmetric weighting for desirable and undesirable examples.

The KTO loss uses per-token cross-entropy scaled by a sign factor:
desirable examples receive positive weight (model should increase prob),
undesirable examples receive negative weight (model should decrease prob).

Since the shared training loop passes (logits, targets) and does not have
access to per-example desirability labels, we encode the signal in the
targets tensor: undesirable examples have all targets negated.
"""

from __future__ import annotations

from typing import Any


def build_kto_loss_function(
    torch_module: Any,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
) -> Any:
    """Build the KTO loss function.

    KTO uses an asymmetric loss: desirable examples are trained with
    standard cross-entropy (weighted), while undesirable examples
    are trained to reduce log-probability (negative gradient).

    We encode desirability in the targets: undesirable targets are
    negated (all set to -1). The loss function detects this and
    flips the gradient direction.
    """
    ce_loss = torch_module.nn.CrossEntropyLoss(reduction="none")

    def kto_loss(logits: Any, targets: Any) -> Any:
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)

        # Detect undesirable examples by checking for negative targets.
        # Undesirable batches have targets set to -1 by _build_kto_batches.
        is_undesirable = flat_targets < 0
        safe_targets = flat_targets.clamp(min=0)

        per_token_loss = ce_loss(flat_logits, safe_targets)

        # Apply asymmetric weighting:
        # desirable: minimize CE (standard) with desirable_weight
        # undesirable: maximize CE (negate loss) with undesirable_weight
        weights = torch_module.where(
            is_undesirable,
            torch_module.tensor(-undesirable_weight * beta, device=per_token_loss.device),
            torch_module.tensor(desirable_weight, device=per_token_loss.device),
        )
        weighted_loss = (per_token_loss * weights).mean()
        return weighted_loss

    return kto_loss
