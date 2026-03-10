"""Custom training loop loader.

This module loads user-provided Python training loop files.
It validates a run_custom_training(context) callable contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, cast

from core.errors import CrucibleServeError
from core.types import EpochMetric
from serve.training_context import TrainingRuntimeContext

CustomLoopCallable = Callable[[TrainingRuntimeContext], list[EpochMetric]]


def load_custom_training_loop(custom_loop_path: str | None) -> CustomLoopCallable | None:
    """Load custom training loop callable when provided.

    Args:
        custom_loop_path: Optional path to Python custom loop file.

    Returns:
        Callable loop or None when path is omitted.

    Raises:
        CrucibleServeError: If path is invalid or callable is missing.
    """
    if custom_loop_path is None:
        return None
    resolved_path = Path(custom_loop_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Custom loop file not found at {resolved_path}. "
            "Provide a valid --custom-loop-file path."
        )
    module = _load_python_module(resolved_path)
    loop_fn = getattr(module, "run_custom_training", None)
    if loop_fn is None or not callable(loop_fn):
        raise CrucibleServeError(
            f"Invalid custom loop file at {resolved_path}: "
            "missing callable run_custom_training(context)."
        )
    return cast(CustomLoopCallable, loop_fn)


def _load_python_module(module_path: Path) -> Any:
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location("crucible_user_training_loop", str(module_path))
    if spec is None or spec.loader is None:
        raise CrucibleServeError(
            f"Failed to load custom loop module at {module_path}. Verify the file path and syntax."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
