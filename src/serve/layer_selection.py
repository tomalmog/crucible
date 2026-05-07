"""Shared layer-index parsing for interpretability runners."""

from __future__ import annotations

from collections.abc import Sequence

from core.errors import CrucibleServeError


def resolve_layer_selection(
    all_layers: Sequence[str],
    layer_indices: str,
) -> list[tuple[int, str]]:
    """Resolve comma-separated layer indexes and ranges into layer names."""
    if not layer_indices.strip():
        return [(index, name) for index, name in enumerate(all_layers)]
    selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    for raw_part in layer_indices.split(","):
        raw = raw_part.strip()
        if not raw:
            continue
        for index in _expand_part(raw):
            if 0 <= index < len(all_layers) and index not in seen:
                selected.append((index, all_layers[index]))
                seen.add(index)
    if not selected:
        raise CrucibleServeError(
            f"No valid layer indices found. Model has {len(all_layers)} layers, "
            f"requested: {layer_indices}"
        )
    return selected


def _expand_part(raw: str) -> range:
    if "-" not in raw:
        index = int(raw)
        return range(index, index + 1)
    start_raw, end_raw = raw.split("-", 1)
    start = int(start_raw.strip())
    end = int(end_raw.strip())
    if end < start:
        raise CrucibleServeError(f"Invalid descending layer range: {raw}")
    return range(start, end + 1)
