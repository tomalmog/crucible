"""Distributed data sampling for DDP training.

This module partitions training batches across distributed ranks so each
GPU processes a unique data slice. It supports uneven splits, assigning
remainder batches to lower-ranked processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedPartition:
    """Partition of batch indices for one rank.

    Attributes:
        rank: Process rank this partition belongs to.
        world_size: Total number of distributed processes.
        total_batches: Total batch count before partitioning.
        start_index: Inclusive start index into the batch list.
        end_index: Exclusive end index into the batch list.
    """

    rank: int
    world_size: int
    total_batches: int
    start_index: int
    end_index: int


def partition_batches_for_rank(
    total_batches: int,
    rank: int,
    world_size: int,
) -> DistributedPartition:
    """Compute batch index range for a given rank.

    Distributes batches as evenly as possible. Lower-ranked processes
    receive one extra batch when the total is not evenly divisible.

    Args:
        total_batches: Number of batches to distribute.
        rank: Target process rank.
        world_size: Total number of processes.

    Returns:
        Partition with start and end indices for this rank.
    """
    per_rank = total_batches // world_size
    remainder = total_batches % world_size
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    return DistributedPartition(
        rank=rank,
        world_size=world_size,
        total_batches=total_batches,
        start_index=start,
        end_index=end,
    )


def select_rank_batches(
    batches: list[Any],
    partition: DistributedPartition,
) -> list[Any]:
    """Select the batch slice assigned to this rank.

    Args:
        batches: Full list of batches.
        partition: Partition describing this rank's slice.

    Returns:
        Sublist of batches for this rank.
    """
    return batches[partition.start_index : partition.end_index]
