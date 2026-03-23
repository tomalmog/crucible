"""Handler for ``crucible remote cluster-info`` CLI sub-subcommand.

Connects to a remote Slurm cluster via SSH, runs ``sinfo`` to gather
partition and node data, and emits a structured JSON summary of GPU
availability, node health, and partition details.
"""

from __future__ import annotations

import argparse
import json as json_mod
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from store.dataset_sdk import CrucibleClient

# Slurm node states that indicate a healthy, schedulable node
_HEALTHY_STATES = frozenset({"idle", "mixed", "allocated", "completing"})
_DRAINED_STATES = frozenset({"drained", "draining"})
_DOWN_STATES = frozenset({"down", "down*", "error", "fail", "not_responding"})

# Match GPU count from GRES field, e.g. "gpu:4(S:0-1)" → 4, "gpu:a100:2" → 2
_GPU_COUNT_RE = re.compile(r"gpu(?::[^:]+)?:(\d+)")

SINFO_FORMAT = "%P|%a|%l|%D|%T|%G|%m|%c"


@dataclass(frozen=True)
class _PartitionRow:
    """Single row from sinfo output, one per partition+state combination."""

    partition: str
    is_default: bool
    avail: str
    time_limit: str
    node_count: int
    state: str
    gres: str
    memory_mb: int
    cpus: int


def _parse_sinfo_line(line: str) -> _PartitionRow | None:
    """Parse one pipe-delimited sinfo output line into a _PartitionRow."""
    parts = line.strip().split("|")
    if len(parts) < 8:
        return None
    raw_partition = parts[0]
    is_default = raw_partition.endswith("*")
    partition = raw_partition.rstrip("*")
    try:
        node_count = int(parts[3])
        memory_mb = int(parts[6])
        cpus = int(parts[7])
    except ValueError:
        return None
    return _PartitionRow(
        partition=partition,
        is_default=is_default,
        avail=parts[1],
        time_limit=parts[2],
        node_count=node_count,
        state=parts[4].lower(),
        gres=parts[5],
        memory_mb=memory_mb,
        cpus=cpus,
    )


def _extract_gpu_count(gres: str) -> int:
    """Extract per-node GPU count from a GRES string like 'gpu:4(S:0-1)'."""
    match = _GPU_COUNT_RE.search(gres)
    return int(match.group(1)) if match else 0


def _classify_state(state: str) -> str:
    """Map a Slurm node state string to a simplified category."""
    base = state.rstrip("*+~#!%$@^-")
    if base in _HEALTHY_STATES:
        return "healthy"
    if base in _DRAINED_STATES:
        return "drained"
    if base in _DOWN_STATES:
        return "down"
    return "unknown"


def _node_state_bucket(state: str) -> str:
    """Map Slurm state to one of the six UI buckets."""
    base = state.rstrip("*+~#!%$@^-")
    if base == "idle":
        return "idle"
    if base == "mixed":
        return "mixed"
    if base in ("allocated", "completing"):
        return "allocated"
    if base in _DRAINED_STATES:
        return "drained"
    if base in _DOWN_STATES:
        return "down"
    return "unknown"


def _build_cluster_info(
    cluster_name: str,
    rows: list[_PartitionRow],
) -> dict[str, object]:
    """Aggregate sinfo rows into a ClusterInfo dict for JSON output."""
    partitions: dict[str, dict[str, object]] = {}

    for row in rows:
        if row.partition not in partitions:
            partitions[row.partition] = {
                "name": row.partition,
                "isDefault": row.is_default,
                "state": row.avail,
                "timeLimit": row.time_limit,
                "totalNodes": 0,
                "nodesByState": {
                    "idle": 0, "mixed": 0, "allocated": 0,
                    "drained": 0, "down": 0, "unknown": 0,
                },
                "totalGpus": 0,
                "idleGpus": 0,
                "gpuConfig": "",
                "memoryMb": row.memory_mb,
                "cpusPerNode": row.cpus,
            }
        part = partitions[row.partition]
        part["totalNodes"] = int(part["totalNodes"]) + row.node_count  # type: ignore[arg-type]
        bucket = _node_state_bucket(row.state)
        nbs = part["nodesByState"]
        nbs[bucket] = nbs[bucket] + row.node_count  # type: ignore[index,operator]

        gpus_per_node = _extract_gpu_count(row.gres)
        if gpus_per_node > 0 and not part["gpuConfig"]:
            part["gpuConfig"] = row.gres
        part["totalGpus"] = int(part["totalGpus"]) + gpus_per_node * row.node_count  # type: ignore[arg-type]
        # Conservative: only idle nodes contribute idle GPUs
        if bucket == "idle":
            part["idleGpus"] = int(part["idleGpus"]) + gpus_per_node * row.node_count  # type: ignore[arg-type]

    partition_list = list(partitions.values())
    total_gpus = sum(int(p["totalGpus"]) for p in partition_list)  # type: ignore[arg-type]
    idle_gpus = sum(int(p["idleGpus"]) for p in partition_list)  # type: ignore[arg-type]
    utilization = round((1 - idle_gpus / total_gpus) * 100, 1) if total_gpus > 0 else 0.0

    healthy = 0
    drained = 0
    down = 0
    total_nodes = 0
    # Use a set of (partition, state) to avoid double-counting shared nodes,
    # but sinfo rows already partition by state — aggregate per-partition totals
    for part in partition_list:
        nbs = part["nodesByState"]
        healthy += int(nbs["idle"]) + int(nbs["mixed"]) + int(nbs["allocated"])  # type: ignore[index,arg-type]
        drained += int(nbs["drained"])  # type: ignore[index,arg-type]
        down += int(nbs["down"]) + int(nbs["unknown"])  # type: ignore[index,arg-type]
        total_nodes += int(part["totalNodes"])  # type: ignore[arg-type]

    return {
        "clusterName": cluster_name,
        "isConnected": True,
        "partitions": partition_list,
        "totalGpus": total_gpus,
        "idleGpus": idle_gpus,
        "gpuUtilizationPct": utilization,
        "healthyNodes": healthy,
        "drainedNodes": drained,
        "downNodes": down,
        "totalNodes": total_nodes,
        "fetchedAt": datetime.now(tz=timezone.utc).isoformat(),
    }


def _disconnected_result(cluster_name: str) -> dict[str, object]:
    """Return a ClusterInfo dict representing a failed SSH connection."""
    return {
        "clusterName": cluster_name,
        "isConnected": False,
        "partitions": [],
        "totalGpus": 0,
        "idleGpus": 0,
        "gpuUtilizationPct": 0.0,
        "healthyNodes": 0,
        "drainedNodes": 0,
        "downNodes": 0,
        "totalNodes": 0,
        "fetchedAt": datetime.now(tz=timezone.utc).isoformat(),
    }


def _handle_cluster_info(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Fetch live cluster state via sinfo and emit CRUCIBLE_JSON."""
    from core.errors import CrucibleRemoteError
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster_name: str = args.cluster
    cluster = load_cluster(client._config.data_root, cluster_name)

    try:
        with SshSession(cluster) as session:
            stdout, stderr, exit_code = session.execute(
                f"sinfo --noheader --format='{SINFO_FORMAT}'",
                timeout=15,
            )
    except CrucibleRemoteError as err:
        print(f"SSH connection failed: {err}", flush=True)
        print("CRUCIBLE_JSON:" + json_mod.dumps(_disconnected_result(cluster_name)))
        return 0

    if exit_code != 0:
        print(f"sinfo exited with code {exit_code}: {stderr.strip()}", flush=True)
        print("CRUCIBLE_JSON:" + json_mod.dumps(_disconnected_result(cluster_name)))
        return 0

    rows: list[_PartitionRow] = []
    for line in stdout.strip().splitlines():
        row = _parse_sinfo_line(line)
        if row is not None:
            rows.append(row)

    result = _build_cluster_info(cluster_name, rows) if rows else _disconnected_result(cluster_name)
    print("CRUCIBLE_JSON:" + json_mod.dumps(result))
    return 0
