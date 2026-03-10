"""Shared run-spec execution engine for CLI and SDK workflows.

This module maps validated run-spec steps to client operations so different
entry points can execute one declarative pipeline path without drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from core.chat_types import ChatOptions, ChatResult
from core.constants import (
    DEFAULT_CHAT_MAX_NEW_TOKENS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_QUALITY_MODEL,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
)
from core.distillation_types import DistillationOptions
from core.domain_adaptation_types import DomainAdaptationOptions
from core.dpo_types import DpoOptions
from core.errors import CrucibleRunSpecError
from core.run_spec import RunSpec, RunSpecStep, load_run_spec
from core.run_spec_fields import (
    float_with_default,
    int_with_default,
    optional_bool,
    optional_int,
    optional_string,
    parse_position_embedding_type,
    required_string,
)
from core.run_spec_option_builders import build_training_options_for_run_spec
from core.rlhf_types import RlhfOptions
from core.run_spec_step_builders import (
    execute_distill_step,
    execute_domain_adapt_step,
    execute_dpo_train_step,
    execute_rlhf_train_step,
    execute_sft_train_step,
)
from core.sft_types import SftOptions
from core.types import IngestOptions, TrainingOptions, TrainingRunResult


class RunSpecDatasetHandle(Protocol):
    """Dataset operations required by run-spec step execution."""

    def export_training(
        self,
        output_dir: str,
        shard_size: int = 1000,
        include_metadata: bool = False,
    ) -> str: ...


class RunSpecClient(Protocol):
    """Client API contract required by run-spec execution."""

    def with_data_root(self, data_root: str) -> Any: ...

    def ingest(self, options: IngestOptions) -> str: ...

    def dataset(self, dataset_name: str) -> Any: ...

    def train(self, options: TrainingOptions) -> TrainingRunResult: ...

    def sft_train(self, options: SftOptions) -> TrainingRunResult: ...

    def dpo_train(self, options: DpoOptions) -> TrainingRunResult: ...

    def rlhf_train(self, options: RlhfOptions) -> TrainingRunResult: ...

    def distill(self, options: DistillationOptions) -> TrainingRunResult: ...

    def domain_adapt(self, options: DomainAdaptationOptions) -> TrainingRunResult: ...

    def chat(self, options: ChatOptions) -> ChatResult: ...

    def hardware_profile(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class RunSpecExecutionContext:
    """In-memory context used to execute run-spec steps."""

    client: RunSpecClient
    default_dataset_name: str | None


def execute_run_spec_file(client: RunSpecClient, spec_file: str) -> tuple[str, ...]:
    """Load and execute a run-spec file, returning printable output lines."""
    spec = load_run_spec(spec_file)
    return execute_run_spec(client, spec)


def execute_run_spec(client: RunSpecClient, spec: RunSpec) -> tuple[str, ...]:
    """Execute a parsed run-spec object and return output lines."""
    execution_client = (
        client.with_data_root(spec.defaults.data_root) if spec.defaults.data_root else client
    )
    context = RunSpecExecutionContext(
        client=execution_client,
        default_dataset_name=spec.defaults.dataset_name,
    )
    output_lines: list[str] = []
    for step in spec.steps:
        output_lines.extend(_execute_step(context, step))
    return tuple(output_lines)


def _execute_step(context: RunSpecExecutionContext, step: RunSpecStep) -> tuple[str, ...]:
    if step.command == "ingest":
        return (_execute_ingest_step(context, step),)
    if step.command == "train":
        return _execute_train_step(context, step)
    if step.command == "sft-train":
        return execute_sft_train_step(context, step)
    if step.command == "dpo-train":
        return execute_dpo_train_step(context, step)
    if step.command == "rlhf-train":
        return execute_rlhf_train_step(context, step)
    if step.command == "distill":
        return execute_distill_step(context, step)
    if step.command == "domain-adapt":
        return execute_domain_adapt_step(context, step)
    if step.command == "export-training":
        return (_execute_export_training_step(context, step),)
    if step.command == "chat":
        return (_execute_chat_step(context, step),)
    if step.command == "hardware-profile":
        return _execute_hardware_profile_step(context)
    raise CrucibleRunSpecError(f"Unsupported run-spec command '{step.command}'.")


def _execute_ingest_step(context: RunSpecExecutionContext, step: RunSpecStep) -> str:
    options = IngestOptions(
        dataset_name=_resolve_dataset_name(context, step),
        source_uri=required_string(step.args, "source"),
        resume=optional_bool(step.args, "resume", default_value=False),
        quality_model=optional_string(step.args, "quality_model") or DEFAULT_QUALITY_MODEL,
    )
    return context.client.ingest(options)


def _execute_train_step(
    context: RunSpecExecutionContext,
    step: RunSpecStep,
) -> tuple[str, ...]:
    result = context.client.train(
        build_training_options_for_run_spec(
            args=step.args,
            dataset_name=_resolve_dataset_name(context, step),
        )
    )
    return (
        f"model_path={result.model_path}",
        f"history_path={result.history_path}",
        f"plot_path={result.plot_path or '-'}",
        f"epochs_completed={result.epochs_completed}",
        f"checkpoint_dir={result.checkpoint_dir or '-'}",
        f"best_checkpoint_path={result.best_checkpoint_path or '-'}",
        f"resumed_from_checkpoint={result.resumed_from_checkpoint or '-'}",
        f"run_id={result.run_id or '-'}",
        f"artifact_contract_path={result.artifact_contract_path or '-'}",
    )


def _execute_export_training_step(
    context: RunSpecExecutionContext,
    step: RunSpecStep,
) -> str:
    dataset = context.client.dataset(_resolve_dataset_name(context, step))
    return cast(
        str,
        dataset.export_training(
            output_dir=required_string(step.args, "output_dir"),
            shard_size=int_with_default(step.args, "shard_size", 1000),
            include_metadata=optional_bool(step.args, "include_metadata", default_value=False),
        ),
    )


def _execute_chat_step(context: RunSpecExecutionContext, step: RunSpecStep) -> str:
    options = ChatOptions(
        dataset_name=_resolve_dataset_name(context, step),
        model_path=required_string(step.args, "model_path"),
        prompt=required_string(step.args, "prompt"),
        architecture_path=optional_string(step.args, "architecture_file"),
        max_new_tokens=int_with_default(step.args, "max_new_tokens", DEFAULT_CHAT_MAX_NEW_TOKENS),
        max_token_length=int_with_default(step.args, "max_token_length", DEFAULT_MAX_TOKEN_LENGTH),
        temperature=float_with_default(step.args, "temperature", DEFAULT_CHAT_TEMPERATURE),
        top_k=int_with_default(step.args, "top_k", DEFAULT_CHAT_TOP_K),
        hidden_dim=int_with_default(step.args, "hidden_dim", DEFAULT_TRAIN_HIDDEN_DIM),
        num_layers=int_with_default(step.args, "num_layers", DEFAULT_TRAIN_NUM_LAYERS),
        attention_heads=int_with_default(
            step.args, "attention_heads", DEFAULT_TRAIN_ATTENTION_HEADS
        ),
        mlp_hidden_dim=int_with_default(step.args, "mlp_hidden_dim", DEFAULT_TRAIN_MLP_HIDDEN_DIM),
        mlp_layers=int_with_default(step.args, "mlp_layers", DEFAULT_TRAIN_MLP_LAYERS),
        dropout=float_with_default(step.args, "dropout", DEFAULT_TRAIN_DROPOUT),
        position_embedding_type=parse_position_embedding_type(step.args),
        vocabulary_size=optional_int(step.args, "vocabulary_size"),
    )
    return context.client.chat(options).response_text


def _execute_hardware_profile_step(context: RunSpecExecutionContext) -> tuple[str, ...]:
    profile = context.client.hardware_profile()
    return tuple(f"{key}={profile[key]}" for key in sorted(profile.keys()))


def _resolve_dataset_name(context: RunSpecExecutionContext, step: RunSpecStep) -> str:
    dataset_name = optional_string(step.args, "dataset")
    if dataset_name:
        return dataset_name
    if context.default_dataset_name:
        return context.default_dataset_name
    raise CrucibleRunSpecError(
        f"Run-spec command '{step.command}' requires dataset. "
        "Set 'dataset' on the step or in top-level defaults."
    )
