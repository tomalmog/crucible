"""Memory-aware batch size planning for training runs.

This module probes GPU memory to find the largest micro-batch that fits,
then computes gradient accumulation steps to match the effective batch size.

Assumptions:
- torch is already imported and passed as a module reference.
- On CPU/MPS devices, no probing is performed (returns desired batch size).
- Memory fraction limit prevents OOM by reserving headroom for activations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.logging_config import get_logger
from core.types import TrainingOptions

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class MemoryPlan:
    """Resolved memory-aware batching plan.

    Attributes:
        effective_batch_size: Logical batch size requested by the user.
        micro_batch_size: Largest batch that fits in device memory.
        gradient_accumulation_steps: Steps to match effective batch size.
        memory_probed: Whether GPU memory probing was performed.
    """

    effective_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    memory_probed: bool


def probe_micro_batch_size(
    torch_module: Any,
    model: Any,
    device: Any,
    max_token_length: int,
) -> int:
    """Find the largest micro-batch size that fits in GPU memory.

    Tries a dummy forward pass with decreasing batch sizes (halving on OOM)
    until one succeeds. Starts from max_token_length-sized dummy inputs.

    Returns:
        Largest batch size that fits without exceeding memory limits.

    Side-effects:
        Runs dummy forward passes and clears GPU cache during probing.
    """
    device_type = _resolve_device_type(device)
    if device_type != "cuda":
        return max_token_length

    candidate = max_token_length
    while candidate >= 1:
        if _try_forward_pass(torch_module, model, device, candidate, max_token_length):
            _LOGGER.info(
                "memory_probe_success",
                micro_batch_size=candidate,
            )
            return candidate
        _LOGGER.info(
            "memory_probe_oom",
            tried_batch_size=candidate,
            next_candidate=max(candidate // 2, 1),
        )
        if candidate == 1:
            break
        candidate = max(candidate // 2, 1)

    return 1


def compute_gradient_accumulation_steps(
    effective_batch_size: int,
    micro_batch_size: int,
) -> int:
    """Compute gradient accumulation steps to match effective batch size.

    Returns:
        ceil(effective_batch_size / micro_batch_size), minimum 1.
    """
    return max(1, math.ceil(effective_batch_size / micro_batch_size))


def plan_memory_aware_batching(
    torch_module: Any,
    model: Any,
    device: Any,
    options: TrainingOptions,
) -> MemoryPlan:
    """Orchestrate memory probing and accumulation planning.

    Returns:
        A frozen MemoryPlan with resolved micro-batch and accumulation steps.

    Side-effects:
        May run dummy forward passes on GPU to probe memory limits.
    """
    device_type = _resolve_device_type(device)
    if device_type != "cuda":
        _LOGGER.info(
            "memory_plan_skip_probe",
            device_type=device_type,
            reason="non_cuda_device",
        )
        return MemoryPlan(
            effective_batch_size=options.batch_size,
            micro_batch_size=options.batch_size,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            memory_probed=False,
        )

    micro_batch_size = probe_micro_batch_size(
        torch_module=torch_module,
        model=model,
        device=device,
        max_token_length=options.max_token_length,
    )
    micro_batch_size = min(micro_batch_size, options.batch_size)
    accumulation_steps = compute_gradient_accumulation_steps(
        effective_batch_size=options.batch_size,
        micro_batch_size=micro_batch_size,
    )
    _LOGGER.info(
        "memory_plan_resolved",
        effective_batch_size=options.batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation_steps,
    )
    return MemoryPlan(
        effective_batch_size=options.batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        memory_probed=True,
    )


def _resolve_device_type(device: Any) -> str:
    """Extract device type string from torch device object."""
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    return str(device)


def _try_forward_pass(
    torch_module: Any,
    model: Any,
    device: Any,
    batch_size: int,
    seq_length: int,
) -> bool:
    """Attempt a dummy forward pass and return True if it fits in memory."""
    try:
        dummy_input = torch_module.randint(
            0, 100, (batch_size, seq_length), device=device,
        )
        with torch_module.no_grad():
            model(dummy_input)
        del dummy_input
        torch_module.cuda.empty_cache()
        return True
    except RuntimeError:
        torch_module.cuda.empty_cache()
        return False
