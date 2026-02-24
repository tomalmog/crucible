"""Unit tests for latency profiler."""

from __future__ import annotations

from core.deployment_types import LatencyProfile
from deploy.latency_profiler import format_latency_report


def test_format_latency_report_header() -> None:
    """Report should start with a header and separator line."""
    profiles = (
        LatencyProfile(
            batch_size=1,
            sequence_length=32,
            device="cpu",
            mean_latency_ms=5.0,
            p50_ms=4.8,
            p95_ms=6.1,
            p99_ms=7.0,
            throughput_tokens_per_sec=6400.0,
        ),
    )
    lines = format_latency_report(profiles)
    assert len(lines) == 3  # header + separator + 1 data row
    assert "Batch" in lines[0]
    assert "SeqLen" in lines[0]
    assert "-" in lines[1]


def test_format_latency_report_data_row() -> None:
    """Data rows should contain the profile values."""
    profiles = (
        LatencyProfile(
            batch_size=4,
            sequence_length=128,
            device="cpu",
            mean_latency_ms=10.5,
            p50_ms=9.8,
            p95_ms=15.2,
            p99_ms=18.1,
            throughput_tokens_per_sec=48762.0,
        ),
    )
    lines = format_latency_report(profiles)
    data_line = lines[2]
    assert "10.50" in data_line
    assert "9.80" in data_line
    assert "48762.0" in data_line


def test_format_latency_report_multiple_profiles() -> None:
    """Report should contain one data line per profile."""
    profiles = (
        LatencyProfile(
            batch_size=1, sequence_length=32, device="cpu",
            mean_latency_ms=5.0, p50_ms=4.8, p95_ms=6.1,
            p99_ms=7.0, throughput_tokens_per_sec=6400.0,
        ),
        LatencyProfile(
            batch_size=8, sequence_length=512, device="cpu",
            mean_latency_ms=25.0, p50_ms=24.0, p95_ms=30.0,
            p99_ms=35.0, throughput_tokens_per_sec=163840.0,
        ),
    )
    lines = format_latency_report(profiles)
    # header + separator + 2 data rows
    assert len(lines) == 4
