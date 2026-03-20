"""Knowledge distillation loss with temperature-scaled KL divergence.

This module computes the blended distillation loss combining soft
teacher-student KL divergence with hard cross-entropy on ground truth
labels, controlled by a temperature and alpha mixing coefficient.
"""

from __future__ import annotations

from typing import Any


def soften_logits(torch_module: Any, logits: Any, temperature: float) -> Any:
    """Apply temperature scaling and log-softmax to logits.

    Args:
        torch_module: Imported torch module.
        logits: Raw logits tensor of shape (batch, seq_len, vocab).
        temperature: Temperature for softening (higher = softer).

    Returns:
        Log-softmax of scaled logits along the vocabulary dimension.
    """
    return torch_module.nn.functional.log_softmax(
        logits / temperature, dim=-1
    )


def _compute_soft_targets(
    torch_module: Any, teacher_logits: Any, temperature: float,
) -> Any:
    """Compute soft target probabilities from teacher logits.

    Args:
        torch_module: Imported torch module.
        teacher_logits: Teacher model logits of shape (batch, seq, vocab).
        temperature: Softening temperature.

    Returns:
        Softmax probability distribution over vocabulary.
    """
    return torch_module.nn.functional.softmax(
        teacher_logits / temperature, dim=-1
    )


def compute_distillation_loss(
    torch_module: Any,
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    temperature: float,
    alpha: float,
) -> Any:
    """Compute blended distillation loss.

    Combines temperature-scaled KL divergence between student and teacher
    with standard cross-entropy on ground truth labels.

    soft_loss = KL(softmax(student/T), softmax(teacher/T)) * T^2
    hard_loss = CrossEntropy(student, labels)
    loss = alpha * soft_loss + (1 - alpha) * hard_loss

    Args:
        torch_module: Imported torch module.
        student_logits: Student logits (batch, seq_len, vocab).
        teacher_logits: Teacher logits (batch, seq_len, vocab).
        labels: Ground truth label indices (batch, seq_len).
        temperature: Softening temperature.
        alpha: Blend coefficient in [0, 1]; 1 = pure KL, 0 = pure CE.

    Returns:
        Scalar loss tensor.
    """
    soft_loss = _compute_kl_loss(
        torch_module, student_logits, teacher_logits, temperature,
    )
    hard_loss = _compute_hard_loss(
        torch_module, student_logits, labels,
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def _compute_kl_loss(
    torch_module: Any,
    student_logits: Any,
    teacher_logits: Any,
    temperature: float,
) -> Any:
    """Compute KL divergence loss scaled by temperature squared.

    When teacher and student have different vocabulary sizes, both are
    sliced to the smaller vocab so KL divergence can be computed over
    the shared token dimensions.

    Args:
        torch_module: Imported torch module.
        student_logits: Student logits (batch, seq_len, vocab).
        teacher_logits: Teacher logits (batch, seq_len, vocab).
        temperature: Softening temperature.

    Returns:
        Scalar KL divergence loss times T^2.
    """
    student_vocab = student_logits.shape[-1]
    teacher_vocab = teacher_logits.shape[-1]
    if student_vocab != teacher_vocab:
        shared_vocab = min(student_vocab, teacher_vocab)
        student_logits = student_logits[..., :shared_vocab]
        teacher_logits = teacher_logits[..., :shared_vocab]
    student_log_probs = soften_logits(torch_module, student_logits, temperature)
    teacher_probs = _compute_soft_targets(torch_module, teacher_logits, temperature)
    kl_loss = torch_module.nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean",
    )
    return kl_loss * (temperature ** 2)


def _compute_hard_loss(
    torch_module: Any, student_logits: Any, labels: Any,
) -> Any:
    """Compute cross-entropy loss on ground truth labels.

    Args:
        torch_module: Imported torch module.
        student_logits: Student logits (batch, seq_len, vocab).
        labels: Ground truth label indices (batch, seq_len).

    Returns:
        Scalar cross-entropy loss.
    """
    batch_size = student_logits.shape[0]
    seq_len = student_logits.shape[1]
    vocab_size = student_logits.shape[2]
    flat_logits = student_logits.reshape(batch_size * seq_len, vocab_size)
    flat_labels = labels.reshape(batch_size * seq_len)
    return torch_module.nn.functional.cross_entropy(flat_logits, flat_labels)
