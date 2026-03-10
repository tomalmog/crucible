"""Unit tests for server database module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from server.database import DatabaseConfig, create_engine_from_config


def test_database_config_is_frozen() -> None:
    """DatabaseConfig should be an immutable frozen dataclass."""
    config = DatabaseConfig()
    with pytest.raises(FrozenInstanceError):
        config.database_url = "sqlite:///other.db"  # type: ignore[misc]


def test_database_config_default_url() -> None:
    """DatabaseConfig should default to SQLite file URL."""
    config = DatabaseConfig()
    assert config.database_url == "sqlite:///crucible_server.db"


def test_create_engine_returns_engine() -> None:
    """create_engine_from_config should return a valid SQLAlchemy engine."""
    config = DatabaseConfig(database_url="sqlite:///:memory:")
    engine = create_engine_from_config(config)
    assert engine is not None
    assert hasattr(engine, "connect")
