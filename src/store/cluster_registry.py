"""CRUD operations for Slurm cluster configurations.

Persists ClusterConfig records as JSON files under .forge/clusters/.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig
from serve.training_run_io import read_json_file, write_json_file


def _clusters_dir(data_root: Path) -> Path:
    """Return the clusters storage directory, creating it if needed."""
    d = data_root / "clusters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cluster_path(data_root: Path, name: str) -> Path:
    """Return the JSON file path for a named cluster."""
    return _clusters_dir(data_root) / f"{name}.json"


def _cluster_to_dict(cluster: ClusterConfig) -> dict[str, object]:
    """Serialize a ClusterConfig to a JSON-safe dictionary."""
    return {
        "name": cluster.name,
        "host": cluster.host,
        "user": cluster.user,
        "ssh_key_path": cluster.ssh_key_path,
        "password": cluster.password,
        "default_partition": cluster.default_partition,
        "partitions": list(cluster.partitions),
        "gpu_types": list(cluster.gpu_types),
        "module_loads": list(cluster.module_loads),
        "python_path": cluster.python_path,
        "remote_workspace": cluster.remote_workspace,
        "exclude_nodes": cluster.exclude_nodes,
        "validated_at": cluster.validated_at,
    }


def _dict_to_cluster(raw: dict[str, object]) -> ClusterConfig:
    """Reconstruct a ClusterConfig from a dictionary."""
    return ClusterConfig(
        name=str(raw["name"]),
        host=str(raw["host"]),
        user=str(raw["user"]),
        ssh_key_path=str(raw.get("ssh_key_path", "")),
        password=str(raw.get("password", "")),
        default_partition=str(raw.get("default_partition", "")),
        partitions=tuple(raw.get("partitions", ())),  # type: ignore[arg-type]
        gpu_types=tuple(raw.get("gpu_types", ())),  # type: ignore[arg-type]
        module_loads=tuple(raw.get("module_loads", ())),  # type: ignore[arg-type]
        python_path=str(raw.get("python_path", "python3")),
        remote_workspace=str(raw.get("remote_workspace", "/tmp/forge-jobs")),
        exclude_nodes=str(raw.get("exclude_nodes", "")),
        validated_at=str(raw.get("validated_at", "")),
    )


def save_cluster(data_root: Path, cluster: ClusterConfig) -> Path:
    """Persist a cluster configuration to disk.

    Args:
        data_root: Root .forge directory.
        cluster: Cluster configuration to save.

    Returns:
        Path to the written JSON file.
    """
    target = _cluster_path(data_root, cluster.name)
    try:
        write_json_file(target, _cluster_to_dict(cluster))
    except Exception as error:
        raise ForgeRemoteError(
            f"Failed to save cluster {cluster.name}: {error}."
        ) from error
    return target


def load_cluster(data_root: Path, name: str) -> ClusterConfig:
    """Load a cluster configuration from disk.

    Args:
        data_root: Root .forge directory.
        name: Cluster name to load.

    Returns:
        Loaded ClusterConfig instance.

    Raises:
        ForgeRemoteError: If the cluster file does not exist or is invalid.
    """
    target = _cluster_path(data_root, name)
    if not target.exists():
        raise ForgeRemoteError(f"Cluster '{name}' not found.")
    try:
        raw = read_json_file(target)
    except Exception as error:
        raise ForgeRemoteError(
            f"Failed to load cluster {name}: {error}."
        ) from error
    if not isinstance(raw, dict):
        raise ForgeRemoteError(f"Invalid cluster data for {name}.")
    return _dict_to_cluster(raw)


def list_clusters(data_root: Path) -> tuple[ClusterConfig, ...]:
    """List all registered cluster configurations.

    Args:
        data_root: Root .forge directory.

    Returns:
        Tuple of ClusterConfig records sorted by name.
    """
    clusters_dir = _clusters_dir(data_root)
    configs: list[ClusterConfig] = []
    for path in sorted(clusters_dir.glob("*.json")):
        try:
            raw = read_json_file(path)
            if isinstance(raw, dict):
                configs.append(_dict_to_cluster(raw))
        except Exception:
            continue
    return tuple(configs)


def remove_cluster(data_root: Path, name: str) -> None:
    """Remove a cluster configuration from disk.

    Args:
        data_root: Root .forge directory.
        name: Cluster name to remove.

    Raises:
        ForgeRemoteError: If the cluster file does not exist.
    """
    target = _cluster_path(data_root, name)
    if not target.exists():
        raise ForgeRemoteError(f"Cluster '{name}' not found.")
    target.unlink()
