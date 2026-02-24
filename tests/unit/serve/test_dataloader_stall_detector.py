"""Unit tests for data loading stall detection."""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError

import pytest

from serve.dataloader_stall_detector import DataloaderStallDetector, StallEvent


def test_no_stall_when_fast_processing() -> None:
    """No stall should be detected when batches complete quickly."""
    detector = DataloaderStallDetector(threshold_seconds=1.0)

    detector.record_batch_start()
    event = detector.record_batch_end(batch_index=0)

    detector.record_batch_start()
    event = detector.record_batch_end(batch_index=1)

    assert event is None
    assert len(detector.stall_events) == 0
    assert detector.total_stall_seconds == 0.0


def test_stall_detected_when_slow() -> None:
    """A stall should be detected when inter-batch gap exceeds threshold."""
    detector = DataloaderStallDetector(threshold_seconds=0.01)

    # Complete first batch
    detector.record_batch_start()
    detector.record_batch_end(batch_index=0)

    # Simulate a slow gap before the next batch starts
    time.sleep(0.05)

    detector.record_batch_start()
    event = detector.record_batch_end(batch_index=1)

    assert event is not None
    assert event.batch_index == 1
    assert event.stall_duration_seconds >= 0.01
    assert len(detector.stall_events) == 1


def test_total_stall_seconds_accumulates() -> None:
    """Total stall seconds should sum all detected stall durations."""
    detector = DataloaderStallDetector(threshold_seconds=0.01)

    # First batch - no stall possible (no previous end time)
    detector.record_batch_start()
    detector.record_batch_end(batch_index=0)

    # Second batch with stall
    time.sleep(0.05)
    detector.record_batch_start()
    detector.record_batch_end(batch_index=1)

    # Third batch with stall
    time.sleep(0.05)
    detector.record_batch_start()
    detector.record_batch_end(batch_index=2)

    assert len(detector.stall_events) == 2
    assert detector.total_stall_seconds >= 0.02


def test_stall_event_frozen() -> None:
    """StallEvent should be immutable."""
    event = StallEvent(batch_index=5, stall_duration_seconds=2.0)

    with pytest.raises(FrozenInstanceError):
        event.batch_index = 10  # type: ignore[misc]
