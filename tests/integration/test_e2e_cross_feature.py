"""Integration tests verifying multiple features work together without conflict."""

from __future__ import annotations

import json
from pathlib import Path

from serve.dataset_curator import score_examples
from serve.synthetic_data import (
    generate_synthetic_data,
    filter_by_quality,
    export_synthetic_data,
)


def test_curate_identifies_quality() -> None:
    """Score a mix of records and verify quality differentiation."""
    records = [
        {
            "record_id": "good1",
            "text": (
                "A well-written document with substantial content"
                " about machine learning and neural networks."
            ),
        },
        {
            "record_id": "good2",
            "text": (
                "Another quality document discussing the fundamentals"
                " of deep learning architectures."
            ),
        },
        {"record_id": "bad1", "text": "hi"},
        {"record_id": "bad2", "text": ""},
    ]
    scores = score_examples(records)
    high_quality = [s for s in scores if s.score >= 0.8]
    low_quality = [s for s in scores if s.score < 0.8]

    assert len(high_quality) >= 1  # at least the good records
    assert len(low_quality) >= 1  # at least the bad ones


def test_synthetic_full_pipeline(tmp_path: Path) -> None:
    """Generate, filter, export, and read back synthetic data."""
    examples = generate_synthetic_data(
        ["Tell me about AI", "What is ML?"], count=10
    )
    assert len(examples) == 10

    filtered = filter_by_quality(examples, min_quality=0.3)
    assert all(e.quality_score >= 0.3 for e in filtered)

    output = str(tmp_path / "synthetic.jsonl")
    count = export_synthetic_data(filtered, output)
    assert count == len(filtered)

    # Read back and verify structure
    lines = Path(output).read_text().strip().splitlines()
    assert len(lines) == count
    for line in lines:
        obj = json.loads(line)
        assert "prompt" in obj
        assert "response" in obj
