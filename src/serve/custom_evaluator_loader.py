"""Custom evaluator function loader.

This module loads user-provided Python evaluation function files.
It validates an evaluate(model, dataset, device) callable contract
that returns a dict mapping metric names to float values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from core.errors import CrucibleServeError

EVALUATOR_FUNCTION_NAME = "evaluate"


def load_custom_evaluator(evaluator_path: str) -> Callable[..., dict[str, float]]:
    """Load custom evaluation function from a .py file.

    Args:
        evaluator_path: Path to Python file containing an evaluate function.

    Returns:
        The validated evaluate callable.

    Raises:
        CrucibleServeError: If path is invalid, function is missing, or validation fails.
    """
    resolved_path = Path(evaluator_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Evaluator file not found at {resolved_path}. "
            "Provide a valid --evaluator-file path."
        )
    module = _load_python_module(resolved_path)
    evaluator_fn = _find_evaluator_function(module, resolved_path)
    validate_evaluator_callable(evaluator_fn)
    return evaluator_fn


def validate_evaluator_callable(evaluator_fn: Any) -> None:
    """Validate that the evaluator is callable.

    Args:
        evaluator_fn: The object to validate as callable.

    Raises:
        CrucibleServeError: If the evaluator is not callable.
    """
    if not callable(evaluator_fn):
        raise CrucibleServeError(
            f"'{EVALUATOR_FUNCTION_NAME}' in evaluator file is not callable."
        )


def _find_evaluator_function(
    module: Any, source_path: Path
) -> Callable[..., dict[str, float]]:
    """Find a function named evaluate in the loaded module."""
    evaluator_fn = getattr(module, EVALUATOR_FUNCTION_NAME, None)
    if evaluator_fn is None:
        raise CrucibleServeError(
            f"Invalid evaluator file at {source_path}: "
            f"missing function '{EVALUATOR_FUNCTION_NAME}'."
        )
    return evaluator_fn


def _load_python_module(module_path: Path) -> Any:
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location(
        "crucible_user_custom_evaluator", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise CrucibleServeError(
            f"Failed to load evaluator module at {module_path}. "
            "Verify the file path and syntax."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
