"""Unit tests for distributed data sampling and batch partitioning."""

from __future__ import annotations

from serve.distributed_data_sampler import (
    DistributedPartition,
    partition_batches_for_rank,
    select_rank_batches,
)


def test_partition_even_split_two_ranks() -> None:
    """Even split across 2 ranks should produce equal halves."""
    p0 = partition_batches_for_rank(total_batches=10, rank=0, world_size=2)
    p1 = partition_batches_for_rank(total_batches=10, rank=1, world_size=2)

    assert p0.start_index == 0 and p0.end_index == 5
    assert p1.start_index == 5 and p1.end_index == 10


def test_partition_even_split_four_ranks() -> None:
    """Even split across 4 ranks should produce 4 equal slices."""
    partitions = [
        partition_batches_for_rank(total_batches=12, rank=r, world_size=4)
        for r in range(4)
    ]

    sizes = [p.end_index - p.start_index for p in partitions]
    assert sizes == [3, 3, 3, 3]


def test_partition_uneven_split_remainder_goes_to_lower_ranks() -> None:
    """When not evenly divisible, lower ranks get one extra batch."""
    p0 = partition_batches_for_rank(total_batches=7, rank=0, world_size=3)
    p1 = partition_batches_for_rank(total_batches=7, rank=1, world_size=3)
    p2 = partition_batches_for_rank(total_batches=7, rank=2, world_size=3)

    assert (p0.end_index - p0.start_index) == 3  # 7 // 3 + 1
    assert (p1.end_index - p1.start_index) == 2  # 7 // 3
    assert (p2.end_index - p2.start_index) == 2  # 7 // 3


def test_partition_covers_all_batches() -> None:
    """All partitions combined should cover exactly all batches."""
    total = 11
    world = 3
    partitions = [
        partition_batches_for_rank(total_batches=total, rank=r, world_size=world)
        for r in range(world)
    ]
    covered = sum(p.end_index - p.start_index for p in partitions)

    assert covered == total


def test_partition_no_overlap() -> None:
    """Partitions should not overlap."""
    total = 10
    world = 3
    partitions = [
        partition_batches_for_rank(total_batches=total, rank=r, world_size=world)
        for r in range(world)
    ]

    for i in range(len(partitions) - 1):
        assert partitions[i].end_index == partitions[i + 1].start_index


def test_partition_single_rank() -> None:
    """Single rank should get all batches."""
    p = partition_batches_for_rank(total_batches=5, rank=0, world_size=1)

    assert p.start_index == 0 and p.end_index == 5


def test_partition_zero_batches() -> None:
    """Zero batches should produce empty partition."""
    p = partition_batches_for_rank(total_batches=0, rank=0, world_size=2)

    assert p.start_index == 0 and p.end_index == 0


def test_partition_preserves_metadata() -> None:
    """Partition should carry rank, world_size, and total_batches."""
    p = partition_batches_for_rank(total_batches=8, rank=1, world_size=4)

    assert p.rank == 1
    assert p.world_size == 4
    assert p.total_batches == 8


def test_select_rank_batches_returns_correct_slice() -> None:
    """select_rank_batches should return the sublist for the partition."""
    batches = ["a", "b", "c", "d", "e"]
    partition = DistributedPartition(
        rank=1, world_size=2, total_batches=5, start_index=2, end_index=5,
    )

    result = select_rank_batches(batches, partition)

    assert result == ["c", "d", "e"]


def test_select_rank_batches_empty_partition() -> None:
    """select_rank_batches should return empty list for empty partition."""
    batches = ["a", "b", "c"]
    partition = DistributedPartition(
        rank=0, world_size=1, total_batches=0, start_index=0, end_index=0,
    )

    result = select_rank_batches(batches, partition)

    assert result == []
