"""Dataset curation utilities.

This module provides quality scoring, distribution analysis,
and data management for training datasets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityScore:
    """Quality assessment for a data record.

    Attributes:
        record_id: Record identifier.
        score: Quality score (0.0 to 1.0).
        issues: List of detected quality issues.
    """

    record_id: str
    score: float
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetDistribution:
    """Distribution statistics for a dataset.

    Attributes:
        total_records: Total number of records.
        avg_token_length: Average token count per record.
        min_token_length: Minimum token count.
        max_token_length: Maximum token count.
        token_length_histogram: Bucket counts for token lengths.
        quality_distribution: Bucket counts for quality scores.
    """

    total_records: int
    avg_token_length: float
    min_token_length: int
    max_token_length: int
    token_length_histogram: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)


def score_examples(records: list[dict[str, Any]]) -> list[QualityScore]:
    """Score quality of data records.

    Detects short content, excessive repetition, and formatting issues.
    """
    results: list[QualityScore] = []
    for i, record in enumerate(records):
        text = record.get("text", record.get("response", ""))
        record_id = record.get("id", str(i))
        issues: list[str] = []
        score = 1.0
        if len(text) < 10:
            issues.append("too_short")
            score -= 0.4
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues.append("highly_repetitive")
                score -= 0.3
        if text.count("\n\n\n") > 2:
            issues.append("excessive_whitespace")
            score -= 0.1
        if not text.strip():
            issues.append("empty_content")
            score = 0.0
        results.append(QualityScore(
            record_id=record_id,
            score=max(0.0, min(1.0, score)),
            issues=tuple(issues),
        ))
    return results


def compute_distributions(records: list[dict[str, Any]]) -> DatasetDistribution:
    """Compute token length and quality distribution statistics."""
    lengths: list[int] = []
    for record in records:
        text = record.get("text", record.get("response", ""))
        lengths.append(len(text.split()))
    if not lengths:
        return DatasetDistribution(
            total_records=0, avg_token_length=0.0,
            min_token_length=0, max_token_length=0,
        )
    avg = sum(lengths) / len(lengths)
    buckets = {"0-50": 0, "50-100": 0, "100-200": 0, "200-500": 0, "500+": 0}
    for length in lengths:
        if length < 50:
            buckets["0-50"] += 1
        elif length < 100:
            buckets["50-100"] += 1
        elif length < 200:
            buckets["100-200"] += 1
        elif length < 500:
            buckets["200-500"] += 1
        else:
            buckets["500+"] += 1
    scores = score_examples(records)
    quality_buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in scores:
        if s.score < 0.2:
            quality_buckets["0.0-0.2"] += 1
        elif s.score < 0.4:
            quality_buckets["0.2-0.4"] += 1
        elif s.score < 0.6:
            quality_buckets["0.4-0.6"] += 1
        elif s.score < 0.8:
            quality_buckets["0.6-0.8"] += 1
        else:
            quality_buckets["0.8-1.0"] += 1
    return DatasetDistribution(
        total_records=len(records),
        avg_token_length=round(avg, 1),
        min_token_length=min(lengths),
        max_token_length=max(lengths),
        token_length_histogram=buckets,
        quality_distribution=quality_buckets,
    )
