"""Training memory estimation.

This module estimates peak GPU memory requirements for a given
training configuration before the training run starts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated memory usage for a training configuration.

    Attributes:
        model_memory_gb: Memory for model parameters.
        optimizer_memory_gb: Memory for optimizer states.
        activation_memory_gb: Memory for activations/gradients.
        total_memory_gb: Total estimated peak memory.
        fits_in_vram: Whether the config fits in available VRAM.
        available_vram_gb: Available GPU VRAM.
    """

    model_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    fits_in_vram: bool
    available_vram_gb: float


def estimate_model_memory(
    hidden_dim: int,
    num_layers: int,
    attention_heads: int,
    vocab_size: int = 32000,
    precision_bytes: int = 4,
) -> float:
    """Estimate model parameter memory in GB."""
    embedding_params = vocab_size * hidden_dim
    head_dim = hidden_dim // max(attention_heads, 1)
    attention_params = 4 * hidden_dim * hidden_dim * num_layers
    ffn_params = 8 * hidden_dim * hidden_dim * num_layers
    total_params = embedding_params + attention_params + ffn_params
    return (total_params * precision_bytes) / (1024**3)


def estimate_optimizer_memory(
    model_memory_gb: float,
    optimizer_type: str = "adamw",
) -> float:
    """Estimate optimizer state memory in GB."""
    multipliers = {"adam": 2.0, "adamw": 2.0, "sgd": 0.0}
    return model_memory_gb * multipliers.get(optimizer_type, 2.0)


def estimate_activation_memory(
    batch_size: int,
    max_token_length: int,
    hidden_dim: int,
    num_layers: int,
    precision_bytes: int = 4,
) -> float:
    """Estimate activation/gradient memory in GB."""
    per_layer = batch_size * max_token_length * hidden_dim * precision_bytes
    total = per_layer * num_layers * 2
    return total / (1024**3)


def estimate_training_memory(
    hidden_dim: int,
    num_layers: int,
    attention_heads: int,
    batch_size: int,
    max_token_length: int,
    optimizer_type: str = "adamw",
    precision_mode: str = "fp32",
    available_vram_gb: float = 0.0,
    vocab_size: int = 32000,
) -> MemoryEstimate:
    """Estimate total training memory requirements."""
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "auto": 4}.get(precision_mode, 4)
    model_mem = estimate_model_memory(
        hidden_dim, num_layers, attention_heads, vocab_size, precision_bytes,
    )
    opt_mem = estimate_optimizer_memory(model_mem, optimizer_type)
    act_mem = estimate_activation_memory(
        batch_size, max_token_length, hidden_dim, num_layers, precision_bytes,
    )
    total = model_mem + opt_mem + act_mem
    fits = available_vram_gb <= 0 or total <= available_vram_gb * 0.9
    return MemoryEstimate(
        model_memory_gb=round(model_mem, 3),
        optimizer_memory_gb=round(opt_mem, 3),
        activation_memory_gb=round(act_mem, 3),
        total_memory_gb=round(total, 3),
        fits_in_vram=fits,
        available_vram_gb=available_vram_gb,
    )
