"""Data transfer helpers for remote Slurm job submission.

Handles uploading agent bundles, configs, scripts, datasets, and
catalogs to remote clusters.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from core.constants import CATALOG_FILE_NAME, DATASETS_DIR_NAME, VERSIONS_DIR_NAME
from core.errors import ForgeRemoteError
from core.slurm_types import (
    ClusterConfig,
    DataStrategy,
    SlurmResourceConfig,
)
from core.training_methods import DATA_PATH_FIELDS
from serve.slurm_script_gen import (
    generate_multi_node_script,
    generate_single_node_script,
)
from serve.ssh_connection import SshSession


def _upload_bundle(session: SshSession, tarball: Path, workdir: str) -> None:
    """Upload the agent tarball to the remote workspace."""
    session.upload(tarball, f"{workdir}/forge-agent.tar.gz")


def _upload_config(
    session: SshSession,
    config: dict[str, object],
    workdir: str,
    filename: str = "training_config.json",
) -> None:
    """Write and upload a training config JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(config, f, indent=2)
        tmp_path = Path(f.name)
    try:
        session.upload(tmp_path, f"{workdir}/{filename}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _upload_script(session: SshSession, script: str, workdir: str) -> None:
    """Write and upload an sbatch script."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False,
    ) as f:
        f.write(script)
        tmp_path = Path(f.name)
    try:
        session.upload(tmp_path, f"{workdir}/job.sh")
    finally:
        tmp_path.unlink(missing_ok=True)


def _submit_sbatch(session: SshSession, workdir: str) -> str:
    """Submit the sbatch script and parse the Slurm job ID."""
    stdout, stderr, code = session.execute(
        f"sbatch {workdir}/job.sh", timeout=30,
    )
    if code != 0:
        raise ForgeRemoteError(f"sbatch failed: {stderr.strip()}")
    # sbatch output: "Submitted batch job 12345"
    parts = stdout.strip().split()
    if len(parts) < 4:
        raise ForgeRemoteError(f"Unexpected sbatch output: {stdout.strip()}")
    return parts[-1]


def _generate_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
) -> str:
    """Select and generate the appropriate sbatch script."""
    if resources.nodes > 1:
        return generate_multi_node_script(
            cluster, resources, job_id, training_method,
        )
    return generate_single_node_script(
        cluster, resources, job_id, training_method,
    )


_RECORD_BASED_METHODS = frozenset({"train", "distill", "domain-adapt"})


def _upload_raw_dataset(
    session: SshSession,
    ds_dir: Path,
    dataset_name: str,
    workdir: str,
    method_args: dict[str, object],
) -> None:
    """Upload raw dataset files and set raw_data_path for remote ingest.

    Used when the dataset directory has no catalog (not yet ingested).
    Uploads the raw files and sets ``raw_data_path`` so the remote
    agent will auto-ingest them before training.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".tar.gz", delete=False,
    ) as f:
        tmp_tar = Path(f.name)
    with tarfile.open(tmp_tar, "w:gz") as tar:
        tar.add(str(ds_dir), arcname=dataset_name)
    session.mkdir_p(f"{workdir}/data")
    session.upload(tmp_tar, f"{workdir}/data/dataset.tar.gz")
    session.execute(
        f"cd {workdir}/data && tar xzf dataset.tar.gz"
        " && rm dataset.tar.gz",
    )
    tmp_tar.unlink(missing_ok=True)
    remote_data = f"{workdir}/data/{dataset_name}"
    method_args["raw_data_path"] = remote_data


def _upload_dataset_catalog(
    session: SshSession,
    data_root: Path,
    dataset_name: str,
    workdir: str,
) -> None:
    """Upload dataset catalog and latest version to remote.

    Reads the local catalog, builds a minimal single-version catalog,
    tars the latest version directory (manifest + records), and
    extracts into ``{workdir}/.forge/datasets/{name}/``.

    Args:
        session: Active SSH session.
        data_root: Local ``.forge`` root directory.
        dataset_name: Dataset identifier.
        workdir: Remote job working directory.
    """
    from store.catalog_io import read_catalog_file

    ds_dir = data_root / DATASETS_DIR_NAME / dataset_name
    catalog_path = ds_dir / CATALOG_FILE_NAME
    catalog = read_catalog_file(catalog_path)

    latest_id = catalog.get("latest_version") or ""
    if not latest_id:
        raise ForgeRemoteError(
            f"Dataset '{dataset_name}' catalog has no latest version.",
        )

    version_dir = ds_dir / VERSIONS_DIR_NAME / latest_id
    if not version_dir.is_dir():
        raise ForgeRemoteError(
            f"Latest version directory missing: {version_dir}",
        )

    # Find the catalog entry for the latest version
    latest_entry = None
    for entry in catalog.get("versions", []):
        if entry.get("version_id") == latest_id:
            latest_entry = entry
            break
    if not latest_entry:
        raise ForgeRemoteError(
            f"Version '{latest_id}' not found in catalog entries.",
        )

    # Build minimal catalog with only the latest version
    minimal_catalog = {
        "latest_version": latest_id,
        "versions": [latest_entry],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Write minimal catalog
        catalog_tmp = tmp_path / CATALOG_FILE_NAME
        catalog_tmp.write_text(
            json.dumps(minimal_catalog, indent=2) + "\n",
            encoding="utf-8",
        )

        # Create tarball: catalog.json + versions/{id}/*
        tar_path = tmp_path / "dataset_catalog.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(catalog_tmp), arcname=CATALOG_FILE_NAME)
            for child in version_dir.iterdir():
                # Skip Lance dir — remote only needs JSONL
                if child.name == "data.lance":
                    continue
                tar.add(
                    str(child),
                    arcname=f"{VERSIONS_DIR_NAME}/{latest_id}/{child.name}",
                )

        remote_ds = f"{workdir}/.forge/{DATASETS_DIR_NAME}/{dataset_name}"
        session.mkdir_p(remote_ds)
        session.upload(tar_path, f"{remote_ds}/dataset_catalog.tar.gz")
        session.execute(
            f"cd {remote_ds} && tar xzf dataset_catalog.tar.gz"
            " && rm dataset_catalog.tar.gz",
        )


def _handle_data_strategy(
    session: SshSession,
    strategy: DataStrategy,
    dataset_path: str,
    method_args: dict[str, object],
    workdir: str,
    training_method: str = "",
    data_root: Path | None = None,
) -> None:
    """Handle data transfer based on the chosen strategy.

    For record-based methods with a ``dataset_name``, uploads the
    dataset catalog and latest version so the remote agent can call
    ``client.train(dataset_name=...)`` directly.

    For ``scp`` strategy, uploads the dataset and rewrites the
    appropriate data path field in *method_args* so the remote
    entry script uses the uploaded path instead of the local one.
    """
    ds_name = str(method_args.get("dataset_name", ""))
    if ds_name and training_method in _RECORD_BASED_METHODS and data_root:
        ds_dir = data_root / DATASETS_DIR_NAME / ds_name
        catalog_path = ds_dir / CATALOG_FILE_NAME
        if catalog_path.exists():
            _upload_dataset_catalog(session, data_root, ds_name, workdir)
        elif ds_dir.is_dir():
            _upload_raw_dataset(
                session, ds_dir, ds_name, workdir, method_args,
            )
        else:
            raise ForgeRemoteError(f"Dataset '{ds_name}' not found.")
        return

    if strategy == "scp" and dataset_path:
        local_data = Path(dataset_path)
        if local_data.is_file():
            remote_data = f"{workdir}/data/{local_data.name}"
            session.mkdir_p(f"{workdir}/data")
            session.upload(local_data, remote_data)
        elif local_data.is_dir():
            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", delete=False,
            ) as f:
                tmp_tar = Path(f.name)
            with tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(str(local_data), arcname=local_data.name)
            session.mkdir_p(f"{workdir}/data")
            session.upload(tmp_tar, f"{workdir}/data/dataset.tar.gz")
            session.execute(
                f"cd {workdir}/data && tar xzf dataset.tar.gz",
            )
            tmp_tar.unlink(missing_ok=True)
            remote_data = f"{workdir}/data/{local_data.name}"
        else:
            return

        # Rewrite data path in method_args to point to uploaded file
        field = DATA_PATH_FIELDS.get(training_method)
        if field:
            method_args[field] = remote_data
        else:
            # train/distill/domain-adapt use raw_data_path for remote
            method_args["raw_data_path"] = remote_data
    # shared and s3 strategies: data path is already in method_args
