"""Custom trainer class loader.

This module loads user-provided Python trainer class files.
It validates that the class exposes a train(self, context) method contract.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from core.errors import CrucibleServeError

TRAINER_CLASS_NAME = "Trainer"
REQUIRED_METHOD = "train"


def load_custom_trainer(trainer_path: str) -> Any:
    """Load and validate a custom trainer class from a .py file.

    Args:
        trainer_path: Path to Python file containing a Trainer class.

    Returns:
        The validated Trainer class (not an instance).

    Raises:
        CrucibleServeError: If path is invalid, class is missing, or validation fails.
    """
    resolved_path = Path(trainer_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Trainer file not found at {resolved_path}. "
            "Provide a valid --trainer-file path."
        )
    module = _load_python_module(resolved_path)
    trainer_class = _find_trainer_class(module, resolved_path)
    validate_trainer_class(trainer_class)
    return trainer_class


def validate_trainer_class(trainer_class: type) -> None:
    """Check that the trainer class has a required train method.

    Args:
        trainer_class: The class to validate.

    Raises:
        CrucibleServeError: If the class lacks a callable train method.
    """
    train_method = getattr(trainer_class, REQUIRED_METHOD, None)
    if train_method is None or not callable(train_method):
        raise CrucibleServeError(
            f"Trainer class '{trainer_class.__name__}' is missing "
            f"a callable '{REQUIRED_METHOD}' method."
        )


def _find_trainer_class(module: Any, source_path: Path) -> type:
    """Find a class named Trainer in the loaded module."""
    trainer_class = getattr(module, TRAINER_CLASS_NAME, None)
    if trainer_class is None or not inspect.isclass(trainer_class):
        raise CrucibleServeError(
            f"Invalid trainer file at {source_path}: "
            f"missing class '{TRAINER_CLASS_NAME}'."
        )
    return trainer_class


def _load_python_module(module_path: Path) -> Any:
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location(
        "crucible_user_custom_trainer", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise CrucibleServeError(
            f"Failed to load trainer module at {module_path}. "
            "Verify the file path and syntax."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
