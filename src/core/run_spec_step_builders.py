"""Run-spec step builders for advanced training commands.

Extracts SFT, DPO, RLHF, distillation, and domain-adaptation step executors
from run_spec_execution so the dispatcher stays concise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTILLATION_ALPHA,
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_DPO_BETA,
    DEFAULT_DPO_LABEL_SMOOTHING,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_SFT_MASK_PROMPT_TOKENS,
    DEFAULT_SFT_PACKING_ENABLED,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)
from core.distillation_types import DistillationOptions
from core.domain_adaptation_types import DomainAdaptationOptions
from core.dpo_types import DpoOptions
from core.rlhf_types import PpoConfig, RewardModelConfig, RlhfOptions
from core.run_spec_fields import (
    float_with_default,
    int_with_default,
    optional_bool,
    optional_string,
    parse_optimizer_type,
    parse_precision_mode,
    required_string,
)
from core.sft_types import SftOptions
from core.types import TrainingRunResult

if TYPE_CHECKING:
    from core.run_spec import RunSpecStep
    from core.run_spec_execution import RunSpecExecutionContext


def format_training_result(result: TrainingRunResult) -> tuple[str, ...]:
    """Format a TrainingRunResult into printable output lines."""
    return (
        f"model_path={result.model_path}",
        f"history_path={result.history_path}",
        f"plot_path={result.plot_path or '-'}",
        f"epochs_completed={result.epochs_completed}",
        f"checkpoint_dir={result.checkpoint_dir or '-'}",
        f"best_checkpoint_path={result.best_checkpoint_path or '-'}",
        f"run_id={result.run_id or '-'}",
        f"artifact_contract_path={result.artifact_contract_path or '-'}",
    )


def _common_training_args(args: Mapping[str, object]) -> dict[str, Any]:
    """Extract common training args shared across all step builders."""
    return {
        "epochs": int_with_default(args, "epochs", DEFAULT_TRAIN_EPOCHS),
        "learning_rate": float_with_default(args, "learning_rate", DEFAULT_TRAIN_LEARNING_RATE),
        "batch_size": int_with_default(args, "batch_size", DEFAULT_BATCH_SIZE),
        "max_token_length": int_with_default(args, "max_token_length", DEFAULT_MAX_TOKEN_LENGTH),
        "validation_split": float_with_default(
            args, "validation_split", DEFAULT_TRAIN_VALIDATION_SPLIT,
        ),
        "precision_mode": parse_precision_mode(args),
        "optimizer_type": parse_optimizer_type(args),
        "weight_decay": float_with_default(args, "weight_decay", DEFAULT_TRAIN_WEIGHT_DECAY),
        "hidden_dim": int_with_default(args, "hidden_dim", DEFAULT_TRAIN_HIDDEN_DIM),
        "num_layers": int_with_default(args, "num_layers", DEFAULT_TRAIN_NUM_LAYERS),
        "attention_heads": int_with_default(args, "attention_heads", DEFAULT_TRAIN_ATTENTION_HEADS),
        "hooks_path": optional_string(args, "hooks_file"),
        "checkpoint_every_epochs": int_with_default(
            args, "checkpoint_every_epochs", DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        ),
        "save_best_checkpoint": optional_bool(args, "save_best_checkpoint", default_value=True),
        "progress_log_interval_steps": int_with_default(
            args, "progress_log_interval_steps", DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
        ),
    }


def execute_sft_train_step(
    context: RunSpecExecutionContext, step: RunSpecStep,
) -> tuple[str, ...]:
    """Build SftOptions from run-spec step args and execute SFT training."""
    from core.run_spec_execution import _resolve_dataset_name

    common = _common_training_args(step.args)
    options = SftOptions(
        dataset_name=_resolve_dataset_name(context, step),
        output_dir=required_string(step.args, "output_dir"),
        sft_data_path=required_string(step.args, "sft_data_path"),
        mask_prompt_tokens=optional_bool(
            step.args, "mask_prompt_tokens", default_value=DEFAULT_SFT_MASK_PROMPT_TOKENS,
        ),
        packing_enabled=optional_bool(
            step.args, "packing_enabled", default_value=DEFAULT_SFT_PACKING_ENABLED,
        ),
        initial_weights_path=optional_string(step.args, "initial_weights_path"),
        **common,
    )
    return format_training_result(context.client.sft_train(options))


def execute_dpo_train_step(
    context: RunSpecExecutionContext, step: RunSpecStep,
) -> tuple[str, ...]:
    """Build DpoOptions from run-spec step args and execute DPO training."""
    from core.run_spec_execution import _resolve_dataset_name

    common = _common_training_args(step.args)
    options = DpoOptions(
        dataset_name=_resolve_dataset_name(context, step),
        output_dir=required_string(step.args, "output_dir"),
        dpo_data_path=required_string(step.args, "dpo_data_path"),
        beta=float_with_default(step.args, "beta", DEFAULT_DPO_BETA),
        label_smoothing=float_with_default(
            step.args, "label_smoothing", DEFAULT_DPO_LABEL_SMOOTHING,
        ),
        reference_model_path=optional_string(step.args, "reference_model_path"),
        initial_weights_path=optional_string(step.args, "initial_weights_path"),
        **common,
    )
    return format_training_result(context.client.dpo_train(options))


def execute_rlhf_train_step(
    context: RunSpecExecutionContext, step: RunSpecStep,
) -> tuple[str, ...]:
    """Build RlhfOptions from run-spec step args and execute RLHF training."""
    from core.run_spec_execution import _resolve_dataset_name

    common = _common_training_args(step.args)
    reward_config = RewardModelConfig(
        reward_model_path=optional_string(step.args, "reward_model_path"),
        train_reward_model=optional_bool(
            step.args, "train_reward_model", default_value=False,
        ),
        preference_data_path=optional_string(step.args, "preference_data_path"),
    )
    ppo_config = PpoConfig(
        clip_epsilon=float_with_default(step.args, "clip_epsilon", 0.2),
        ppo_epochs=int_with_default(step.args, "ppo_epochs", 4),
        entropy_coeff=float_with_default(step.args, "entropy_coeff", 0.01),
    )
    options = RlhfOptions(
        dataset_name=_resolve_dataset_name(context, step),
        output_dir=required_string(step.args, "output_dir"),
        policy_model_path=required_string(step.args, "policy_model_path"),
        reward_config=reward_config, ppo_config=ppo_config,
        initial_weights_path=optional_string(step.args, "initial_weights_path"),
        **common,
    )
    return format_training_result(context.client.rlhf_train(options))


def execute_distill_step(
    context: RunSpecExecutionContext, step: RunSpecStep,
) -> tuple[str, ...]:
    """Build DistillationOptions from run-spec step args and execute."""
    from core.run_spec_execution import _resolve_dataset_name

    common = _common_training_args(step.args)
    options = DistillationOptions(
        dataset_name=_resolve_dataset_name(context, step),
        output_dir=required_string(step.args, "output_dir"),
        teacher_model_path=required_string(step.args, "teacher_model_path"),
        student_model_path=optional_string(step.args, "student_model_path"),
        temperature=float_with_default(
            step.args, "temperature", DEFAULT_DISTILLATION_TEMPERATURE,
        ),
        alpha=float_with_default(step.args, "alpha", DEFAULT_DISTILLATION_ALPHA),
        initial_weights_path=optional_string(step.args, "initial_weights_path"),
        **common,
    )
    return format_training_result(context.client.distill(options))


def execute_domain_adapt_step(
    context: RunSpecExecutionContext, step: RunSpecStep,
) -> tuple[str, ...]:
    """Build DomainAdaptationOptions from run-spec step args and execute."""
    from core.run_spec_execution import _resolve_dataset_name

    common = _common_training_args(step.args)
    options = DomainAdaptationOptions(
        dataset_name=_resolve_dataset_name(context, step),
        output_dir=required_string(step.args, "output_dir"),
        base_model_path=required_string(step.args, "base_model_path"),
        reference_data_path=optional_string(step.args, "reference_data_path"),
        drift_check_interval_epochs=int_with_default(
            step.args, "drift_check_interval_epochs", 1,
        ),
        max_perplexity_increase=float_with_default(
            step.args, "max_perplexity_increase", 1.5,
        ),
        **common,
    )
    return format_training_result(context.client.domain_adapt(options))
