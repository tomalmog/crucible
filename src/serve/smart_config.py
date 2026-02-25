"""Smart hardware-aware training configuration suggestions.

This module recommends training hyperparameters based on available
hardware, model size, and training method.
"""

from __future__ import annotations

from dataclasses import dataclass

from serve.gpu_profiles import GpuProfile, get_gpu_profile
from serve.memory_estimator import estimate_training_memory


@dataclass(frozen=True)
class TrainingConfigSuggestion:
    """Suggested training configuration based on hardware.

    Attributes:
        batch_size: Recommended batch size.
        precision_mode: Recommended precision mode.
        gradient_accumulation_steps: Recommended gradient accumulation.
        use_qlora: Whether to use QLoRA instead of full fine-tuning.
        estimated_memory_gb: Estimated peak memory usage.
        estimated_time_hours: Rough training time estimate.
        gpu_name: GPU used for the suggestion.
        notes: Additional recommendations.
    """

    batch_size: int
    precision_mode: str
    gradient_accumulation_steps: int
    use_qlora: bool
    estimated_memory_gb: float
    estimated_time_hours: float
    gpu_name: str
    notes: list[str]


def suggest_training_config(
    model_size_billions: float,
    training_method: str,
    gpu_name: str,
    dataset_size_examples: int = 10000,
    target_epochs: int = 3,
) -> TrainingConfigSuggestion:
    """Suggest training configuration based on hardware and model size."""
    profile = get_gpu_profile(gpu_name)
    vram_gb = profile.vram_gb if profile else 16.0
    gpu_display = profile.name if profile else gpu_name
    notes: list[str] = []
    model_memory_gb = model_size_billions * 4.0
    use_qlora = model_memory_gb > vram_gb * 0.5
    if use_qlora:
        effective_model_mem = model_size_billions * 0.5
        notes.append("QLoRA recommended due to model size vs VRAM")
    else:
        effective_model_mem = model_memory_gb
    precision = "bf16"
    if profile and "mps" in profile.compute_capability:
        precision = "fp32"
        notes.append("Apple Silicon: using fp32 (bf16 not fully supported on MPS)")
    elif profile and float(profile.compute_capability.replace("mps", "0")) < 8.0:
        precision = "fp16"
    available_for_batch = vram_gb - effective_model_mem - (effective_model_mem * 0.5)
    batch_size = max(1, int(available_for_batch / 0.5))
    batch_size = min(batch_size, 64)
    grad_accum = 1
    if batch_size < 8:
        grad_accum = max(1, 8 // batch_size)
        notes.append(f"Using gradient accumulation ({grad_accum}x) to achieve effective batch size {batch_size * grad_accum}")
    steps_per_epoch = max(1, dataset_size_examples // (batch_size * grad_accum))
    total_steps = steps_per_epoch * target_epochs
    throughput = profile.fp16_tflops if profile else 10.0
    time_per_step = (model_size_billions * 1e9 * 6) / (throughput * 1e12)
    estimated_hours = (total_steps * time_per_step) / 3600
    mem_estimate = estimate_training_memory(
        hidden_dim=int((model_size_billions * 1e9 / 12) ** 0.5) if model_size_billions > 0 else 256,
        num_layers=int(model_size_billions * 4) if model_size_billions > 0 else 2,
        attention_heads=max(1, int(model_size_billions * 4)),
        batch_size=batch_size, max_token_length=512,
        optimizer_type="adamw", precision_mode=precision,
        available_vram_gb=vram_gb,
    )
    if training_method in ("lora-train", "qlora-train"):
        notes.append("LoRA/QLoRA: only adapter weights are trained, reducing memory significantly")
    if training_method in ("dpo-train", "rlhf-train", "kto-train", "orpo-train"):
        notes.append("Preference methods may need reference model — consider additional memory")
    return TrainingConfigSuggestion(
        batch_size=batch_size, precision_mode=precision,
        gradient_accumulation_steps=grad_accum, use_qlora=use_qlora,
        estimated_memory_gb=mem_estimate.total_memory_gb,
        estimated_time_hours=round(estimated_hours, 2),
        gpu_name=gpu_display, notes=notes,
    )
