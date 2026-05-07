"""Tests for shared interpretability layer selection."""

from __future__ import annotations

import pytest

from core.errors import CrucibleServeError
from serve.layer_selection import resolve_layer_selection


def test_resolve_layer_selection_supports_ranges() -> None:
    layers = ["layer.0", "layer.1", "layer.2", "layer.3"]

    selected = resolve_layer_selection(layers, "0-2")

    assert selected == [(0, "layer.0"), (1, "layer.1"), (2, "layer.2")]


def test_resolve_layer_selection_deduplicates_indexes() -> None:
    layers = ["layer.0", "layer.1", "layer.2", "layer.3"]

    selected = resolve_layer_selection(layers, "0,1-2,2")

    assert selected == [(0, "layer.0"), (1, "layer.1"), (2, "layer.2")]


def test_resolve_layer_selection_rejects_descending_ranges() -> None:
    with pytest.raises(CrucibleServeError):
        resolve_layer_selection(["layer.0", "layer.1"], "1-0")
