"""Data loading stall detection for training diagnostics.

This module wraps batch iteration to detect slow data loading,
helping identify I/O bottlenecks in the training pipeline.

Assumptions:
- The caller invokes record_batch_start() and record_batch_end() around
  each batch processing step.
- A stall is any gap between the end of the previous batch and the start
  of the current batch that exceeds the configured threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class StallEvent:
    """One detected stall in data loading.

    Attributes:
        batch_index: Index of the batch that experienced the stall.
        stall_duration_seconds: Duration of the stall in seconds.
    """

    batch_index: int
    stall_duration_seconds: float


class DataloaderStallDetector:
    """Detects stalls between batch yields during training.

    Wraps batch iteration and flags delays exceeding the threshold.
    """

    def __init__(self, threshold_seconds: float) -> None:
        self._threshold_seconds = threshold_seconds
        self._batch_end_time: float | None = None
        self._batch_start_time: float | None = None
        self._stall_events: list[StallEvent] = []

    def record_batch_start(self) -> None:
        """Mark the start of batch processing.

        Side-effects:
            Updates internal batch-start timestamp.
        """
        self._batch_start_time = time.monotonic()

    def record_batch_end(self, batch_index: int) -> StallEvent | None:
        """Mark end of batch processing and check for stall.

        Returns:
            StallEvent if the gap between batches exceeded threshold,
            else None.

        Side-effects:
            Appends a StallEvent to internal list when a stall is detected.
        """
        now = time.monotonic()
        stall_event = self._check_for_stall(batch_index)
        self._batch_end_time = now
        return stall_event

    @property
    def stall_events(self) -> tuple[StallEvent, ...]:
        """Return all detected stall events."""
        return tuple(self._stall_events)

    @property
    def total_stall_seconds(self) -> float:
        """Return total time spent in detected stalls."""
        return sum(event.stall_duration_seconds for event in self._stall_events)

    def _check_for_stall(self, batch_index: int) -> StallEvent | None:
        """Check whether the gap since last batch end exceeds threshold."""
        if self._batch_end_time is None or self._batch_start_time is None:
            return None
        gap = self._batch_start_time - self._batch_end_time
        if gap < self._threshold_seconds:
            return None
        event = StallEvent(
            batch_index=batch_index,
            stall_duration_seconds=gap,
        )
        self._stall_events.append(event)
        return event
