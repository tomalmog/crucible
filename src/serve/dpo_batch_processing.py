"""DPO batch processing and training loop internals.

Handles the DPO training loop, batch loss computation, and the
runtime context / result types used during DPO training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.dpo_types import DpoOptions
from core.training_types import EpochMetric
from serve.dpo_loss import compute_dpo_loss, compute_log_probs_from_logits
from serve.dpo_reference_model import compute_reference_log_probs
from serve.dpo_tokenization import DpoTokenizedPair
from core.chat_types import ChatTokenizer
from serve.training_progress import emit_progress
from core.types import TrainingOptions


class DpoContext:
    """Internal context for DPO training state."""

    def __init__(
        self,
        torch_module: Any,
        model: Any,
        ref_model: Any,
        tokenizer: ChatTokenizer,
        dpo_pairs: list[DpoTokenizedPair],
        output_dir: Path,
        device: Any,
        training_options: TrainingOptions,
    ) -> None:
        self.torch_module = torch_module
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dpo_pairs = dpo_pairs
        self.output_dir = output_dir
        self.device = device
        self.training_options = training_options


class DpoLoopResult:
    """DPO training loop output."""

    def __init__(self, epoch_metrics: list[EpochMetric]) -> None:
        self.epoch_metrics = epoch_metrics


def run_dpo_loop(context: DpoContext, options: DpoOptions) -> DpoLoopResult:
    """Execute the DPO training loop over epochs and batches."""
    torch_module = context.torch_module
    optimizer = torch_module.optim.Adam(
        context.model.parameters(), lr=options.learning_rate,
    )
    start_epoch = 1
    global_step = 0
    if options.resume_checkpoint_path:
        from serve.training_checkpoint import load_resume_checkpoint
        resume = load_resume_checkpoint(
            options.resume_checkpoint_path, torch_module, context.model,
            optimizer, None, context.device,
        )
        start_epoch = resume.next_epoch
        global_step = resume.global_step
    total_batches = max(1, len(context.dpo_pairs) // options.batch_size)
    emit_progress(
        "training_started",
        total_epochs=options.epochs,
        start_epoch=start_epoch,
        method="dpo",
    )
    epoch_metrics: list[EpochMetric] = []
    epoch = start_epoch
    checkpoint_dir = None
    try:
        for epoch in range(start_epoch, options.epochs + 1):
            context.model.train()
            total_loss = 0.0
            batch_count = 0
            emit_progress(
                "training_epoch_started",
                epoch=epoch,
                total_epochs=options.epochs,
            )
            for batch_start in range(
                0, len(context.dpo_pairs), options.batch_size
            ):
                batch = context.dpo_pairs[
                    batch_start: batch_start + options.batch_size
                ]
                loss = _compute_batch_loss(
                    context, batch, options.beta, options.label_smoothing,
                )
                optimizer.zero_grad()
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    context.model.parameters(), max_norm=1.0,
                )
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                global_step += 1
                emit_progress(
                    "training_batch_progress",
                    epoch=epoch,
                    total_epochs=options.epochs,
                    batch=batch_count,
                    total_batches=total_batches,
                    loss=round(loss.item(), 6),
                )
            avg_loss = total_loss / max(batch_count, 1)
            emit_progress(
                "training_epoch_completed",
                epoch=epoch,
                total_epochs=options.epochs,
                train_loss=round(avg_loss, 6),
            )
            epoch_metrics.append(EpochMetric(
                epoch=epoch,
                train_loss=avg_loss,
                validation_loss=avg_loss,
            ))
            from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
            checkpoint_dir = ensure_checkpoint_dir(Path(context.output_dir))
            save_epoch_checkpoint(
                checkpoint_dir, torch_module, context.model, optimizer, None,
                epoch, global_step, None,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving emergency checkpoint...", flush=True)
        from serve.training_checkpoint import save_epoch_checkpoint, ensure_checkpoint_dir
        emergency_dir = checkpoint_dir or ensure_checkpoint_dir(Path(context.output_dir))
        emergency_path = save_epoch_checkpoint(
            emergency_dir, torch_module, context.model, optimizer, None,
            epoch, global_step, None,
        )
        print(f"Emergency checkpoint saved: {emergency_path}", flush=True)
        raise
    return DpoLoopResult(epoch_metrics=epoch_metrics)


def _compute_batch_loss(
    context: DpoContext,
    batch: list[DpoTokenizedPair],
    beta: float,
    label_smoothing: float,
) -> Any:
    """Compute DPO loss for a single batch of preference pairs."""
    torch_module = context.torch_module
    all_losses = []
    for pair in batch:
        chosen_ids = torch_module.tensor(
            [list(pair.chosen_ids)], device=context.device,
        )
        rejected_ids = torch_module.tensor(
            [list(pair.rejected_ids)], device=context.device,
        )
        chosen_labels = torch_module.tensor(
            [list(pair.chosen_labels)], device=context.device,
        )
        rejected_labels = torch_module.tensor(
            [list(pair.rejected_labels)], device=context.device,
        )
        if getattr(context.model, "_is_hf_logits_wrapper", False):
            chosen_mask = (chosen_ids != 0).long()
            rejected_mask = (rejected_ids != 0).long()
            chosen_logits = context.model(chosen_ids, attention_mask=chosen_mask)
            rejected_logits = context.model(rejected_ids, attention_mask=rejected_mask)
        else:
            chosen_logits = context.model(chosen_ids)
            rejected_logits = context.model(rejected_ids)
        pi_chosen = compute_log_probs_from_logits(
            torch_module, chosen_logits, chosen_labels, pair.prompt_length,
        )
        pi_rejected = compute_log_probs_from_logits(
            torch_module, rejected_logits, rejected_labels,
            pair.prompt_length,
        )
        ref_chosen = compute_reference_log_probs(
            torch_module, context.ref_model, chosen_ids,
            chosen_labels, pair.prompt_length,
        )
        ref_rejected = compute_reference_log_probs(
            torch_module, context.ref_model, rejected_ids,
            rejected_labels, pair.prompt_length,
        )
        loss = compute_dpo_loss(
            torch_module, pi_chosen, pi_rejected,
            ref_chosen, ref_rejected, beta, label_smoothing,
        )
        all_losses.append(loss)
    return torch_module.stack(all_losses).mean()
