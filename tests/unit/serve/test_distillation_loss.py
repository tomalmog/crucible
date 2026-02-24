"""Unit tests for knowledge distillation loss computation."""

from __future__ import annotations

import torch

from serve.distillation_loss import (
    compute_distillation_loss,
    soften_logits,
)


def test_soften_logits_sums_to_one() -> None:
    """Softened logits should produce valid log-probabilities."""
    logits = torch.randn(2, 4, 10)
    temperature = 2.0
    log_probs = soften_logits(torch, logits, temperature)
    probs = log_probs.exp()
    sums = probs.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_higher_temperature_produces_flatter_distribution() -> None:
    """Higher temperature should produce a more uniform distribution."""
    logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])
    low_temp_probs = soften_logits(torch, logits, 1.0).exp()
    high_temp_probs = soften_logits(torch, logits, 10.0).exp()
    low_temp_max = low_temp_probs.max().item()
    high_temp_max = high_temp_probs.max().item()

    assert high_temp_max < low_temp_max


def test_compute_distillation_loss_returns_finite_scalar() -> None:
    """Distillation loss should return a finite scalar tensor."""
    batch_size, seq_len, vocab_size = 2, 4, 10
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_distillation_loss(
        torch, student_logits, teacher_logits, labels,
        temperature=2.0, alpha=0.5,
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_alpha_one_uses_only_kl_divergence() -> None:
    """With alpha=1.0, loss should equal pure KL divergence."""
    batch_size, seq_len, vocab_size = 1, 3, 5
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss_alpha_1 = compute_distillation_loss(
        torch, student_logits, teacher_logits, labels,
        temperature=2.0, alpha=1.0,
    )
    # With alpha=1, changing labels should not affect the loss
    different_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_diff_labels = compute_distillation_loss(
        torch, student_logits, teacher_logits, different_labels,
        temperature=2.0, alpha=1.0,
    )

    assert torch.allclose(loss_alpha_1, loss_diff_labels, atol=1e-5)


def test_alpha_zero_uses_only_cross_entropy() -> None:
    """With alpha=0.0, loss should equal pure cross-entropy."""
    batch_size, seq_len, vocab_size = 1, 3, 5
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss_alpha_0 = compute_distillation_loss(
        torch, student_logits, teacher_logits, labels,
        temperature=2.0, alpha=0.0,
    )
    # With alpha=0, changing teacher logits should not affect loss
    different_teacher = torch.randn(batch_size, seq_len, vocab_size)
    loss_diff_teacher = compute_distillation_loss(
        torch, student_logits, different_teacher, labels,
        temperature=2.0, alpha=0.0,
    )

    assert torch.allclose(loss_alpha_0, loss_diff_teacher, atol=1e-5)


def test_identical_student_teacher_minimizes_kl() -> None:
    """When student matches teacher, KL component should be near zero."""
    batch_size, seq_len, vocab_size = 2, 4, 8
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_distillation_loss(
        torch, logits, logits.clone(), labels,
        temperature=2.0, alpha=1.0,
    )

    assert loss.item() < 1e-5
