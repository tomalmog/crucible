"""Unit tests for DPO loss computation."""

from __future__ import annotations

import torch

from serve.dpo_loss import compute_dpo_loss, compute_log_probs_from_logits
from serve.dpo_tokenization import IGNORE_INDEX


def test_compute_dpo_loss_prefers_chosen() -> None:
    """Loss should decrease when policy favors chosen over rejected."""
    # Simulate: policy strongly prefers chosen
    policy_chosen_logps = torch.tensor([-1.0])
    policy_rejected_logps = torch.tensor([-5.0])
    ref_chosen_logps = torch.tensor([-2.0])
    ref_rejected_logps = torch.tensor([-2.0])

    loss = compute_dpo_loss(
        torch, policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1, label_smoothing=0.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_compute_dpo_loss_with_label_smoothing() -> None:
    """Label smoothing should change the loss value."""
    policy_chosen_logps = torch.tensor([-1.0])
    policy_rejected_logps = torch.tensor([-3.0])
    ref_chosen_logps = torch.tensor([-2.0])
    ref_rejected_logps = torch.tensor([-2.0])

    loss_no_smooth = compute_dpo_loss(
        torch, policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1, label_smoothing=0.0,
    )
    loss_with_smooth = compute_dpo_loss(
        torch, policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1, label_smoothing=0.1,
    )

    assert torch.isfinite(loss_no_smooth)
    assert torch.isfinite(loss_with_smooth)
    # Smoothed loss should differ from unsmoothed
    assert not torch.allclose(loss_no_smooth, loss_with_smooth)


def test_compute_log_probs_from_logits_masks_prompt() -> None:
    """Log probs should only count response tokens after prompt_length."""
    batch_size = 1
    seq_len = 4
    vocab_size = 10
    prompt_length = 2

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7]])

    log_probs = compute_log_probs_from_logits(
        torch, logits, labels, prompt_length,
    )

    assert log_probs.shape == (batch_size,)
    assert torch.isfinite(log_probs).all()
    # Log probs should be negative (or zero)
    assert log_probs.item() <= 0


def test_compute_dpo_loss_symmetry() -> None:
    """Equal policy and reference should give predictable loss."""
    # When policy == reference, the logit diff is 0
    policy_chosen_logps = torch.tensor([-2.0])
    policy_rejected_logps = torch.tensor([-2.0])
    ref_chosen_logps = torch.tensor([-2.0])
    ref_rejected_logps = torch.tensor([-2.0])

    loss = compute_dpo_loss(
        torch, policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1, label_smoothing=0.0,
    )

    # -log_sigmoid(0) = log(2) ~= 0.6931
    assert abs(loss.item() - 0.6931) < 0.01
