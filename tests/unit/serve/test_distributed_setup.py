"""Unit tests for distributed training process group setup."""

from __future__ import annotations

import pytest

from core.errors import CrucibleDistributedError
from serve.distributed_setup import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    select_ddp_backend,
)


class _FakeDist:
    """Mock torch.distributed module."""

    def __init__(self, initialized: bool = False, rank: int = 0, world: int = 1) -> None:
        self._initialized = initialized
        self._rank = rank
        self._world = world
        self.init_called = False
        self.destroy_called = False

    def is_initialized(self) -> bool:
        return self._initialized

    def init_process_group(self, backend: str = "nccl") -> None:
        self.init_called = True
        self._initialized = True

    def destroy_process_group(self) -> None:
        self.destroy_called = True
        self._initialized = False

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """Mock torch module with configurable distributed."""

    def __init__(
        self,
        dist: _FakeDist | None = None,
        cuda_available: bool = False,
    ) -> None:
        self.distributed = dist
        self.cuda = _FakeCuda(cuda_available)

    def device(self, value: str) -> str:
        return value


def test_init_distributed_calls_init_process_group() -> None:
    """init_distributed should call dist.init_process_group."""
    dist = _FakeDist(initialized=False)
    torch_mod = _FakeTorch(dist=dist)

    init_distributed(torch_mod, backend="gloo")

    assert dist.init_called


def test_init_distributed_skips_when_already_initialized() -> None:
    """init_distributed should not re-initialize when already active."""
    dist = _FakeDist(initialized=True)
    torch_mod = _FakeTorch(dist=dist)

    init_distributed(torch_mod, backend="gloo")

    assert not dist.init_called


def test_cleanup_distributed_destroys_group() -> None:
    """cleanup_distributed should destroy the active process group."""
    dist = _FakeDist(initialized=True)
    torch_mod = _FakeTorch(dist=dist)

    cleanup_distributed(torch_mod)

    assert dist.destroy_called


def test_cleanup_distributed_noop_when_not_initialized() -> None:
    """cleanup_distributed should not call destroy when not initialized."""
    dist = _FakeDist(initialized=False)
    torch_mod = _FakeTorch(dist=dist)

    cleanup_distributed(torch_mod)

    assert not dist.destroy_called


def test_get_rank_returns_zero_when_not_initialized() -> None:
    """get_rank should return 0 when process group is not initialized."""
    dist = _FakeDist(initialized=False)
    torch_mod = _FakeTorch(dist=dist)

    assert get_rank(torch_mod) == 0


def test_get_rank_returns_dist_rank_when_initialized() -> None:
    """get_rank should return the distributed rank when initialized."""
    dist = _FakeDist(initialized=True, rank=3)
    torch_mod = _FakeTorch(dist=dist)

    assert get_rank(torch_mod) == 3


def test_get_world_size_returns_one_when_not_initialized() -> None:
    """get_world_size should return 1 when not distributed."""
    dist = _FakeDist(initialized=False)
    torch_mod = _FakeTorch(dist=dist)

    assert get_world_size(torch_mod) == 1


def test_get_world_size_returns_dist_value_when_initialized() -> None:
    """get_world_size should return distributed world size."""
    dist = _FakeDist(initialized=True, world=4)
    torch_mod = _FakeTorch(dist=dist)

    assert get_world_size(torch_mod) == 4


def test_is_main_process_true_for_rank_zero() -> None:
    """is_main_process should return True for rank 0."""
    dist = _FakeDist(initialized=True, rank=0)
    torch_mod = _FakeTorch(dist=dist)

    assert is_main_process(torch_mod) is True


def test_is_main_process_false_for_nonzero_rank() -> None:
    """is_main_process should return False for non-zero ranks."""
    dist = _FakeDist(initialized=True, rank=2)
    torch_mod = _FakeTorch(dist=dist)

    assert is_main_process(torch_mod) is False


def test_raises_when_distributed_unavailable() -> None:
    """Should raise CrucibleDistributedError when torch.distributed is None."""
    torch_mod = _FakeTorch(dist=None)

    with pytest.raises(CrucibleDistributedError, match="not available"):
        get_rank(torch_mod)


def test_select_ddp_backend_prefers_nccl_with_cuda() -> None:
    """select_ddp_backend should return nccl when CUDA is available."""
    torch_mod = _FakeTorch(cuda_available=True)

    assert select_ddp_backend(torch_mod) == "nccl"


def test_select_ddp_backend_falls_back_to_gloo() -> None:
    """select_ddp_backend should return gloo when CUDA is unavailable."""
    torch_mod = _FakeTorch(cuda_available=False)

    assert select_ddp_backend(torch_mod) == "gloo"
