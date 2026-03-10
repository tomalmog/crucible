"""Unit tests for server authentication module."""

from __future__ import annotations

import pytest

from server.auth import (
    create_user,
    generate_api_key,
    require_api_key,
    validate_api_key,
)
from server.database import DatabaseConfig, create_engine_from_config, create_session_factory, init_database
from core.errors import CrucibleServerError


@pytest.fixture()
def db_session():
    """Create an in-memory database session for testing."""
    config = DatabaseConfig(database_url="sqlite:///:memory:")
    engine = create_engine_from_config(config)
    init_database(engine)
    factory = create_session_factory(engine)
    session = factory()
    yield session
    session.close()


def test_generate_api_key_format() -> None:
    """generate_api_key should return a uuid4-format string."""
    key = generate_api_key()
    assert isinstance(key, str)
    assert len(key) == 36
    parts = key.split("-")
    assert len(parts) == 5


def test_validate_api_key_returns_none_for_unknown(db_session) -> None:
    """validate_api_key should return None for non-existent key."""
    result = validate_api_key("nonexistent-key", db_session)
    assert result is None


def test_require_api_key_raises_for_invalid(db_session) -> None:
    """require_api_key should raise CrucibleServerError for invalid key."""
    with pytest.raises(CrucibleServerError, match="Invalid API key"):
        require_api_key("bad-key", db_session)


def test_create_user_and_validate(db_session) -> None:
    """create_user should create a user that can be validated."""
    user, api_key = create_user("testuser", db_session)
    assert user.username == "testuser"
    assert api_key == user.api_key

    found = validate_api_key(api_key, db_session)
    assert found is not None
    assert found.username == "testuser"
