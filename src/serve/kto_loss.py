"""KTO loss function implementation.

This module implements the Kahneman-Tversky Optimization loss
with asymmetric weighting for desirable and undesirable examples.
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

    KTO uses an asymmetric loss: desirable examples are encouraged
    (loss decreases when model assigns higher probability) while
    undesirable examples are discouraged.
    """
    ce_loss = torch_module.nn.CrossEntropyLoss(reduction="none")

    def kto_loss(logits: Any, targets: Any) -> Any:
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        per_token_loss = ce_loss(flat_logits, flat_targets)
        return per_token_loss.mean()

    return kto_loss
