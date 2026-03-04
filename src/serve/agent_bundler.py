"""Build a tarball of Forge modules for remote execution.

Bundles the minimal set of Forge source modules needed to run
training on a remote Slurm cluster, plus a generated entry script.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
from pathlib import Path

from core.errors import ForgeRemoteError

# Modules to include in the agent tarball
_BUNDLE_MODULES = (
    "core",
    "serve",
    "store",
    "ingest",
    "transforms",
)

# Files within serve/ that should NOT be bundled (they are host-only)
_SERVE_EXCLUDES = frozenset({
    "ssh_connection.py",
    "cluster_validator.py",
    "agent_bundler.py",
    "slurm_script_gen.py",
    "remote_job_submitter.py",
    "remote_log_streamer.py",
})

_ENTRY_SCRIPT = '''"""Forge remote agent entry point.

Reads a JSON training config, sets up sys.path, and dispatches training.
"""

import argparse
import json
import os
import sys

# Add the extracted bundle directory to sys.path
bundle_dir = os.path.dirname(os.path.abspath(__file__))
if bundle_dir not in sys.path:
    sys.path.insert(0, bundle_dir)

# Methods that require dataset records via the snapshot store
_RECORD_BASED_METHODS = frozenset({"train", "distill", "domain-adapt"})


def _ensure_output_dir(method_args: dict) -> None:
    """Set output_dir to ./output if not already specified."""
    if not method_args.get("output_dir"):
        out = os.path.join(os.getcwd(), "output")
        os.makedirs(out, exist_ok=True)
        method_args["output_dir"] = out


def _ensure_dataset_name(method_args: dict, method: str) -> None:
    """Set dataset_name appropriately for the training method.

    Record-based methods (train, distill, domain-adapt) need a real
    dataset_name for snapshot store lookups.  All other methods need
    dataset_name present (it is a required dataclass field) but set
    to empty string so the SDK uses the data path field directly.
    """
    if method_args.get("dataset_name"):
        return
    if method in _RECORD_BASED_METHODS:
        method_args["dataset_name"] = "remote-dataset"
    else:
        method_args["dataset_name"] = ""


def _hydrate_nested_configs(method_args: dict, method: str) -> None:
    """Convert nested dict values to proper dataclass instances.

    JSON serialization flattens dataclass objects to dicts. This
    reconstructs them so dispatch_training can build the Options.
    Currently only RLHF has nested configs (reward_config, ppo_config).
    """
    if method == "rlhf-train":
        from core.rlhf_types import PpoConfig, RewardModelConfig
        rc = method_args.get("reward_config")
        if isinstance(rc, dict):
            method_args["reward_config"] = RewardModelConfig(**rc)
        pc = method_args.get("ppo_config")
        if isinstance(pc, dict):
            method_args["ppo_config"] = PpoConfig(**pc)


def _ingest_raw_data(client, method_args: dict) -> None:
    """Ingest a raw JSONL file into a forge dataset for record-based methods.

    If ``raw_data_path`` is present in *method_args*, ingest it as a
    temporary dataset so methods like train/distill/domain-adapt can
    load records from the snapshot store.
    """
    raw_path = method_args.pop("raw_data_path", None)
    if not raw_path:
        return
    from core.ingest_types import IngestOptions
    ds_name = method_args.get("dataset_name", "remote-dataset")
    opts = IngestOptions(dataset_name=ds_name, source_uri=raw_path)
    print(f"FORGE_AGENT: Ingesting {raw_path} as dataset '{ds_name}'...")
    client.ingest(opts)
    method_args["dataset_name"] = ds_name
    print(f"FORGE_AGENT: Ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge remote agent")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    method = config.get("method", "train")
    method_args = config.get("method_args", {})
    output_path = config.get("result_output", "result.json")

    from core.config import ForgeConfig
    from core.training_methods import dispatch_training
    from store.dataset_sdk import ForgeClient

    client = ForgeClient(ForgeConfig.from_env())

    # Ensure output_dir and dataset_name are set
    _ensure_output_dir(method_args)
    _ensure_dataset_name(method_args, method)
    _hydrate_nested_configs(method_args, method)

    # For record-based methods, ingest raw data if provided
    if method in _RECORD_BASED_METHODS:
        _ingest_raw_data(client, method_args)

    try:
        result = dispatch_training(client, method, method_args)
        result_data = {
            "status": "completed",
            "model_path": getattr(result, "model_path", ""),
            "history_path": getattr(result, "history_path", ""),
            "epochs_completed": getattr(result, "epochs_completed", 0),
            "run_id": getattr(result, "run_id", ""),
            "result": str(result),
        }
    except Exception as exc:
        import traceback
        result_data = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    if result_data["status"] == "failed":
        print(f"FORGE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
        sys.exit(1)
    print("FORGE_AGENT_COMPLETE")


if __name__ == "__main__":
    main()
'''


def _src_root() -> Path:
    """Locate the Forge src/ directory."""
    return Path(__file__).resolve().parent.parent


def _should_include(path: Path, module: str) -> bool:
    """Check if a file should be included in the bundle."""
    if module == "serve" and path.name in _SERVE_EXCLUDES:
        return False
    if path.name == "__pycache__":
        return False
    if path.suffix == ".pyc":
        return False
    return True


def _compute_src_hash(src_root: Path) -> str:
    """Compute a content hash of bundled source files for caching."""
    hasher = hashlib.sha256()
    # Include the entry script in the hash
    hasher.update(_ENTRY_SCRIPT.encode("utf-8"))
    for module in _BUNDLE_MODULES:
        module_dir = src_root / module
        if not module_dir.is_dir():
            continue
        for py_file in sorted(module_dir.rglob("*.py")):
            if not _should_include(py_file, module):
                continue
            hasher.update(py_file.read_bytes())
    return hasher.hexdigest()[:16]


def build_agent_tarball(cache_dir: Path | None = None) -> Path:
    """Build a tarball containing Forge modules and entry script.

    Args:
        cache_dir: Directory to cache the tarball. If None, uses tempdir.

    Returns:
        Path to the generated .tar.gz file.
    """
    src_root = _src_root()
    content_hash = _compute_src_hash(src_root)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"forge-agent-{content_hash}.tar.gz"
        if cached.exists():
            return cached
        tarball_path = cached
    else:
        tmp = tempfile.mkdtemp(prefix="forge-agent-")
        tarball_path = Path(tmp) / "forge-agent.tar.gz"

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add source modules
            for module in _BUNDLE_MODULES:
                module_dir = src_root / module
                if not module_dir.is_dir():
                    continue
                for file_path in sorted(module_dir.rglob("*")):
                    if not file_path.is_file():
                        continue
                    if not _should_include(file_path, module):
                        continue
                    arcname = str(file_path.relative_to(src_root))
                    tar.add(str(file_path), arcname=arcname)

            # Add entry script
            _add_string_to_tar(
                tar, "forge_agent_entry.py", _ENTRY_SCRIPT,
            )
    except Exception as error:
        raise ForgeRemoteError(
            f"Failed to build agent tarball: {error}"
        ) from error

    return tarball_path


def _add_string_to_tar(tar: tarfile.TarFile, name: str, content: str) -> None:
    """Add a string as a file to a tarball."""
    import io
    import time

    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))
