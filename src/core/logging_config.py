"""Structured logging configuration.

This module initializes a logger with a stable structured format.
It prefers structlog and falls back to standard logging if absent.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable


@runtime_checkable
class StructuredLogger(Protocol):
    """Protocol for structured loggers (structlog or stdlib fallback)."""

    def debug(self, event: str, **fields: object) -> None: ...
    def info(self, event: str, **fields: object) -> None: ...
    def warning(self, event: str, **fields: object) -> None: ...
    def error(self, event: str, **fields: object) -> None: ...


def get_logger(name: str) -> StructuredLogger:
    """Return a module logger instance.

    Args:
        name: Logger name, usually __name__.

    Returns:
        A structlog or stdlib logger with structured output.
    """
    try:
        import structlog
    except ImportError:
        return _get_standard_logger(name)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)


def _get_standard_logger(name: str) -> _StructuredStandardLogger:
    """Create a stdlib logger fallback.

    Args:
        name: Logger name.

    Returns:
        Configured standard logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return _StructuredStandardLogger(logger)


class _StructuredStandardLogger:
    """Stdlib logger adapter that accepts structured keyword fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, event: str, **fields: object) -> None:
        """Log a debug-level structured event."""
        self._logger.debug(_format_event(event, fields))

    def info(self, event: str, **fields: object) -> None:
        """Log an info-level structured event."""
        self._logger.info(_format_event(event, fields))

    def warning(self, event: str, **fields: object) -> None:
        """Log a warning-level structured event."""
        self._logger.warning(_format_event(event, fields))

    def error(self, event: str, **fields: object) -> None:
        """Log an error-level structured event."""
        self._logger.error(_format_event(event, fields))


def _format_event(event: str, fields: dict[str, object]) -> str:
    """Render a structured event line for standard logging.

    Args:
        event: Event name.
        fields: Event fields.

    Returns:
        JSON-encoded event string.
    """
    if not fields:
        return event
    payload = {"event": event, **fields}
    return json.dumps(payload, sort_keys=True)
