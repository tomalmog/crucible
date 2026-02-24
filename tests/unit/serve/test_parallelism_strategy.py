"""Unit tests for the parallelism strategy protocol and factory."""

from __future__ import annotations

import pytest

from core.errors import ForgeDistributedError
from serve.parallelism_strategy import (
    ParallelismStrategy,
    resolve_parallelism_strategy,
)


def test_resolve_ddp_strategy() -> None:
    """resolve_parallelism_strategy('ddp') should return a DdpStrategy."""
    strategy = resolve_parallelism_strategy("ddp")

    assert strategy.strategy_name() == "ddp"
    assert isinstance(strategy, ParallelismStrategy)


def test_resolve_unknown_raises() -> None:
    """resolve_parallelism_strategy with unknown name should raise."""
    with pytest.raises(ForgeDistributedError, match="Unknown parallelism"):
        resolve_parallelism_strategy("nonexistent_strategy")


def test_protocol_has_required_methods() -> None:
    """ParallelismStrategy protocol should define all required methods."""
    required = {
        "wrap_model",
        "build_optimizer",
        "save_checkpoint",
        "load_checkpoint",
        "strategy_name",
    }
    protocol_methods = {
        name for name in dir(ParallelismStrategy)
        if not name.startswith("_")
    }

    assert required.issubset(protocol_methods)
