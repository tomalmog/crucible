"""Training hook loading and invocation helpers.

This module exposes safe extension points for run-level and loop-level hooks
without requiring users to replace the entire training loop implementation.
"""

from __future__ import annotations

import importlib.util
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from core.errors import CrucibleServeError

# Hook type aliases use Any for the context parameter because of a circular
# import: training_context imports TrainingHooks from this module, so we
# cannot import TrainingRuntimeContext here. At runtime, the first argument
# is always a TrainingRuntimeContext instance.
OnRunStartHook = Callable[[Any], None]
OnEpochStartHook = Callable[[Any, int], None]
OnBatchEndHook = Callable[[Any, str, int, int, int, float], None]
OnEpochEndHook = Callable[[Any, int, float, float], None]
OnCheckpointHook = Callable[[Any, int, str], None]
OnRunEndHook = Callable[[Any, Any], None]
OnRunErrorHook = Callable[[Any, str], None]
BuildLossFunctionHook = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class TrainingHooks:
    """Optional user-defined training hook callables."""

    on_run_start: OnRunStartHook | None = None
    on_epoch_start: OnEpochStartHook | None = None
    on_batch_end: OnBatchEndHook | None = None
    on_epoch_end: OnEpochEndHook | None = None
    on_checkpoint: OnCheckpointHook | None = None
    on_run_end: OnRunEndHook | None = None
    on_run_error: OnRunErrorHook | None = None
    build_loss_function: BuildLossFunctionHook | None = None


def load_training_hooks(hooks_path: str | None) -> TrainingHooks:
    """Load optional hook functions from a Python module path."""
    if hooks_path is None:
        return TrainingHooks()
    resolved_path = Path(hooks_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Hooks file not found at {resolved_path}. Provide a valid --hooks-file path."
        )
    module = _load_python_module(resolved_path)
    return TrainingHooks(
        on_run_start=cast(OnRunStartHook | None, _load_optional_hook(module, "on_run_start")),
        on_epoch_start=cast(
            OnEpochStartHook | None,
            _load_optional_hook(module, "on_epoch_start"),
        ),
        on_batch_end=cast(OnBatchEndHook | None, _load_optional_hook(module, "on_batch_end")),
        on_epoch_end=cast(OnEpochEndHook | None, _load_optional_hook(module, "on_epoch_end")),
        on_checkpoint=cast(
            OnCheckpointHook | None,
            _load_optional_hook(module, "on_checkpoint"),
        ),
        on_run_end=cast(OnRunEndHook | None, _load_optional_hook(module, "on_run_end")),
        on_run_error=cast(OnRunErrorHook | None, _load_optional_hook(module, "on_run_error")),
        build_loss_function=cast(
            BuildLossFunctionHook | None,
            _load_optional_hook(module, "build_loss_function"),
        ),
    )


def build_loss_function_from_hooks(
    torch_module: Any,
    hooks: TrainingHooks,
    runtime_context: Any,
) -> Any:
    """Build loss function from hook module or use default cross-entropy."""
    if hooks.build_loss_function is None:
        return torch_module.nn.CrossEntropyLoss(ignore_index=0)
    try:
        loss_function = hooks.build_loss_function(runtime_context, torch_module)
    except Exception as error:
        raise CrucibleServeError(
            f"Hook 'build_loss_function' failed: {error}. "
            "Fix the hook function or remove --hooks-file."
        ) from error
    if loss_function is None:
        raise CrucibleServeError(
            "Hook 'build_loss_function' returned None. Return a torch loss module instance."
        )
    return loss_function


def invoke_hook(
    hook_name: str,
    hook_fn: Callable[..., object] | None,
    *args: object,
) -> None:
    """Invoke one optional hook and wrap failures with context."""
    if hook_fn is None:
        return
    try:
        hook_fn(*args)
    except Exception as error:
        raise CrucibleServeError(
            f"Hook '{hook_name}' failed: {error}. Fix the hook function or remove --hooks-file."
        ) from error


def _load_optional_hook(module: types.ModuleType, attribute_name: str) -> Callable[..., object] | None:
    hook = getattr(module, attribute_name, None)
    if hook is None:
        return None
    if callable(hook):
        return cast(Callable[..., object], hook)
    raise CrucibleServeError(f"Invalid hooks file: '{attribute_name}' exists but is not callable.")


def _load_python_module(module_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("crucible_user_training_hooks", str(module_path))
    if spec is None or spec.loader is None:
        raise CrucibleServeError(
            f"Failed to load hooks module at {module_path}. Verify file path and syntax."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
