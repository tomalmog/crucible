"""Unit tests for cluster registry CRUD operations."""

from __future__ import annotations

import pytest

from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig
from store.cluster_registry import (
    list_clusters,
    load_cluster,
    remove_cluster,
    save_cluster,
)


def _make_cluster(name: str = "test-hpc", host: str = "hpc.example.com") -> ClusterConfig:
    return ClusterConfig(
        name=name,
        host=host,
        user="testuser",
        ssh_key_path="/home/testuser/.ssh/id_rsa",
        default_partition="gpu",
        partitions=("gpu", "cpu"),
        gpu_types=("a100", "v100"),
        module_loads=("module load cuda/12.1",),
        python_path="python3",
        remote_workspace="/scratch/crucible",
    )


def test_save_and_load_cluster(tmp_path: object) -> None:
    """save_cluster then load_cluster should round-trip a ClusterConfig."""
    cluster = _make_cluster()
    save_cluster(tmp_path, cluster)  # type: ignore[arg-type]
    loaded = load_cluster(tmp_path, "test-hpc")  # type: ignore[arg-type]
    assert loaded.name == cluster.name
    assert loaded.host == cluster.host
    assert loaded.user == cluster.user
    assert loaded.ssh_key_path == cluster.ssh_key_path
    assert loaded.default_partition == "gpu"
    assert loaded.partitions == ("gpu", "cpu")
    assert loaded.gpu_types == ("a100", "v100")
    assert loaded.module_loads == ("module load cuda/12.1",)
    assert loaded.remote_workspace == "/scratch/crucible"


def test_load_missing_cluster_raises(tmp_path: object) -> None:
    """load_cluster should raise CrucibleRemoteError for nonexistent cluster."""
    with pytest.raises(CrucibleRemoteError, match="not found"):
        load_cluster(tmp_path, "nonexistent")  # type: ignore[arg-type]


def test_list_clusters_empty(tmp_path: object) -> None:
    """list_clusters should return empty tuple when no clusters exist."""
    result = list_clusters(tmp_path)  # type: ignore[arg-type]
    assert result == ()


def test_list_clusters_returns_all(tmp_path: object) -> None:
    """list_clusters should return all saved clusters sorted by name."""
    save_cluster(tmp_path, _make_cluster("beta", "beta.example.com"))  # type: ignore[arg-type]
    save_cluster(tmp_path, _make_cluster("alpha", "alpha.example.com"))  # type: ignore[arg-type]
    result = list_clusters(tmp_path)  # type: ignore[arg-type]
    assert len(result) == 2
    assert result[0].name == "alpha"
    assert result[1].name == "beta"


def test_remove_cluster(tmp_path: object) -> None:
    """remove_cluster should delete the cluster file."""
    save_cluster(tmp_path, _make_cluster())  # type: ignore[arg-type]
    remove_cluster(tmp_path, "test-hpc")  # type: ignore[arg-type]
    with pytest.raises(CrucibleRemoteError, match="not found"):
        load_cluster(tmp_path, "test-hpc")  # type: ignore[arg-type]


def test_remove_missing_cluster_raises(tmp_path: object) -> None:
    """remove_cluster should raise CrucibleRemoteError for nonexistent cluster."""
    with pytest.raises(CrucibleRemoteError, match="not found"):
        remove_cluster(tmp_path, "nonexistent")  # type: ignore[arg-type]


def test_save_overwrites_existing(tmp_path: object) -> None:
    """Saving a cluster with the same name should overwrite."""
    save_cluster(tmp_path, _make_cluster("test-hpc", "old.example.com"))  # type: ignore[arg-type]
    save_cluster(tmp_path, _make_cluster("test-hpc", "new.example.com"))  # type: ignore[arg-type]
    loaded = load_cluster(tmp_path, "test-hpc")  # type: ignore[arg-type]
    assert loaded.host == "new.example.com"


def test_cluster_defaults(tmp_path: object) -> None:
    """ClusterConfig with minimal fields should use defaults."""
    cluster = ClusterConfig(name="minimal", host="h", user="u")
    save_cluster(tmp_path, cluster)  # type: ignore[arg-type]
    loaded = load_cluster(tmp_path, "minimal")  # type: ignore[arg-type]
    assert loaded.python_path == "python3"
    assert loaded.remote_workspace == "~/crucible-jobs"
    assert loaded.partitions == ()
    assert loaded.gpu_types == ()
